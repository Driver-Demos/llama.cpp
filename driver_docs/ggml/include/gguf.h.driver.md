# Purpose
This C header file provides a comprehensive interface for handling "GGUF" files, which are a binary file format used by the ggml library. The file defines the structure and operations related to GGUF files, which are designed to store ggml tensors and key-value pairs. The GGUF file format includes a specific structure with a magic number, versioning, and sections for key-value pairs and tensor data, allowing for efficient serialization and deserialization of data. The file includes definitions for various data types that can be stored in GGUF files, such as integers, floats, booleans, strings, and arrays, and provides functions to manipulate these data types within the context of a GGUF file.

The header file defines a public API for initializing, manipulating, and writing GGUF files. It includes functions to initialize GGUF contexts, retrieve and set key-value pairs, manage tensor data, and write the context to a binary file. The API supports operations such as finding keys and tensors, retrieving their types and values, and setting new values or arrays. Additionally, the file provides mechanisms for writing GGUF files in different ways, including writing the entire context or just metadata, and appending tensor data. This functionality is encapsulated in a C interface with provisions for C++ compatibility, making it suitable for integration into larger projects that require efficient handling of ggml tensor data and associated metadata.
# Imports and Dependencies

---
- `ggml.h`
- `stdbool.h`
- `stdint.h`


# Data Structures

---
### gguf\_type
- **Type**: `enum`
- **Members**:
    - `GGUF_TYPE_UINT8`: Represents an unsigned 8-bit integer type.
    - `GGUF_TYPE_INT8`: Represents a signed 8-bit integer type.
    - `GGUF_TYPE_UINT16`: Represents an unsigned 16-bit integer type.
    - `GGUF_TYPE_INT16`: Represents a signed 16-bit integer type.
    - `GGUF_TYPE_UINT32`: Represents an unsigned 32-bit integer type.
    - `GGUF_TYPE_INT32`: Represents a signed 32-bit integer type.
    - `GGUF_TYPE_FLOAT32`: Represents a 32-bit floating point type.
    - `GGUF_TYPE_BOOL`: Represents a boolean type stored as an int8_t.
    - `GGUF_TYPE_STRING`: Represents a string type.
    - `GGUF_TYPE_ARRAY`: Represents an array type.
    - `GGUF_TYPE_UINT64`: Represents an unsigned 64-bit integer type.
    - `GGUF_TYPE_INT64`: Represents a signed 64-bit integer type.
    - `GGUF_TYPE_FLOAT64`: Represents a 64-bit floating point type.
    - `GGUF_TYPE_COUNT`: Marks the end of the enum and is used to count the number of types.
- **Description**: The `gguf_type` enum defines a set of data types that can be used as key-value pair values in GGUF files, which are part of the binary file format used by ggml. Each enumerator represents a specific data type, such as various integer sizes, floating-point numbers, booleans, strings, and arrays, allowing for flexible data representation within the GGUF file structure. The `GGUF_TYPE_COUNT` is a special enumerator used to indicate the total number of types defined in the enum.


---
### gguf\_init\_params
- **Type**: `struct`
- **Members**:
    - `no_alloc`: A boolean flag indicating whether memory allocation should be avoided.
    - `ctx`: A pointer to a pointer of ggml_context, used to create and allocate tensor data if not NULL.
- **Description**: The `gguf_init_params` structure is used to initialize parameters for creating a GGUF context. It contains a boolean flag `no_alloc` to specify whether memory allocation should be avoided, and a pointer `ctx` to a `ggml_context` pointer, which, if not NULL, is used to create a context and allocate tensor data within it. This structure is essential for configuring the initialization behavior of GGUF contexts, particularly in terms of memory management and context creation.


# Function Declarations (Public API)

---
### gguf\_init\_empty<!-- {{#callable_declaration:gguf_init_empty}} -->
Initializes an empty GGUF context.
- **Description**: This function is used to create a new, empty `gguf_context` structure, which is essential for working with GGUF files. It should be called when you need to start a new context for reading or writing GGUF data. There are no specific preconditions for calling this function, and it does not require any parameters. The caller is responsible for managing the memory of the created context, which should be freed using the appropriate function when it is no longer needed.
- **Inputs**: None
- **Output**: Returns a pointer to a newly allocated `gguf_context`. If memory allocation fails, the behavior is undefined, and the caller should ensure that the returned pointer is valid before use.
- **See also**: [`gguf_init_empty`](../src/gguf.cpp.driver.md#gguf_init_empty)  (Implementation)


---
### gguf\_init\_from\_file<!-- {{#callable_declaration:gguf_init_from_file}} -->
Initializes a GGUF context from a file.
- **Description**: This function is used to create a `gguf_context` by reading a GGUF file specified by the `fname` parameter. It is essential to call this function with a valid file path that points to an existing GGUF file. The function will attempt to open the file in binary read mode, and if the file cannot be opened, it will return a null pointer. The `params` argument allows for additional initialization options, such as whether to allocate memory for the context. It is important to ensure that the file adheres to the GGUF format specifications, as any deviations may lead to undefined behavior.
- **Inputs**:
    - `fname`: A pointer to a null-terminated string representing the path to the GGUF file. This must not be null and should point to a valid file that exists on the filesystem.
    - `params`: A `gguf_init_params` structure that contains initialization parameters. The `no_alloc` field specifies whether to allocate memory for the context, and the `ctx` field can be a pointer to a `ggml_context` pointer for additional context management. The structure must be properly initialized before passing it to the function.
- **Output**: Returns a pointer to a `gguf_context` if the initialization is successful. If the file cannot be opened or if there is an error during initialization, it returns a null pointer.
- **See also**: [`gguf_init_from_file`](../src/gguf.cpp.driver.md#gguf_init_from_file)  (Implementation)


---
### gguf\_free<!-- {{#callable_declaration:gguf_free}} -->
Frees the resources associated with a GGUF context.
- **Description**: This function should be called to release the memory allocated for a `gguf_context` when it is no longer needed. It is important to ensure that the context has been properly initialized before calling this function. If the provided context pointer is `NULL`, the function will safely return without performing any action, preventing potential errors. This function does not return any value.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the context to be freed. This pointer must not be null; if it is null, the function will simply return without any action.
- **Output**: None
- **See also**: [`gguf_free`](../src/gguf.cpp.driver.md#gguf_free)  (Implementation)


---
### gguf\_type\_name<!-- {{#callable_declaration:gguf_type_name}} -->
Returns the name of the specified GGUF type.
- **Description**: This function is used to retrieve the string representation of a given `gguf_type` enumeration value. It is particularly useful for debugging or logging purposes, allowing developers to understand what type of data is being handled. The function should be called with a valid `gguf_type` value, which must be within the defined range of the enumeration. If an invalid type is provided, the function will return `NULL`.
- **Inputs**:
    - `type`: An enumeration value of type `gguf_type`. Valid values are from `GGUF_TYPE_UINT8` (0) to `GGUF_TYPE_COUNT` (which marks the end of the enum). Must not be null. If an invalid value is passed, the function will return `NULL`.
- **Output**: Returns a pointer to a string representing the name of the specified `gguf_type`. If the type is invalid, it returns `NULL`.
- **See also**: [`gguf_type_name`](../src/gguf.cpp.driver.md#gguf_type_name)  (Implementation)


---
### gguf\_get\_version<!-- {{#callable_declaration:gguf_get_version}} -->
Retrieves the version of the GGUF context.
- **Description**: This function is used to obtain the version number of the GGUF context, which is essential for ensuring compatibility with the GGUF file format. It should be called after initializing a `gguf_context` to verify the version of the GGUF file being processed. The version is represented as a 32-bit unsigned integer, and it is important to check this value to handle any version-specific features or changes in the file format.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that must be valid and initialized. It must not be null; otherwise, the behavior is undefined.
- **Output**: Returns a 32-bit unsigned integer representing the version of the GGUF context.
- **See also**: [`gguf_get_version`](../src/gguf.cpp.driver.md#gguf_get_version)  (Implementation)


---
### gguf\_get\_alignment<!-- {{#callable_declaration:gguf_get_alignment}} -->
Retrieves the alignment value from the context.
- **Description**: This function is used to obtain the alignment value associated with a `gguf_context`. It should be called after the context has been properly initialized. The alignment value is crucial for ensuring that data structures are aligned in memory according to the specified requirements. If the context is not initialized or is invalid, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure. Must not be null and should point to a valid, initialized context. Passing a null or uninitialized context may lead to undefined behavior.
- **Output**: Returns the alignment value as a `size_t`, which indicates the alignment requirement for data structures in the context.
- **See also**: [`gguf_get_alignment`](../src/gguf.cpp.driver.md#gguf_get_alignment)  (Implementation)


---
### gguf\_get\_data\_offset<!-- {{#callable_declaration:gguf_get_data_offset}} -->
Retrieves the data offset from the context.
- **Description**: This function is used to obtain the current data offset from a `gguf_context`. It should be called after the context has been properly initialized. The returned offset indicates the position in the data where the next read or write operation will occur. It is important to ensure that the `ctx` parameter is valid and not null before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds the state of the GGUF file. Must not be null and should be properly initialized before use.
- **Output**: Returns the current data offset as a `size_t` value, which indicates the position in the data.
- **See also**: [`gguf_get_data_offset`](../src/gguf.cpp.driver.md#gguf_get_data_offset)  (Implementation)


---
### gguf\_get\_n\_kv<!-- {{#callable_declaration:gguf_get_n_kv}} -->
Returns the number of key-value pairs in the context.
- **Description**: This function is used to retrieve the count of key-value pairs stored in a `gguf_context`. It should be called after the context has been properly initialized and populated with data. The function will return a non-negative integer representing the number of key-value pairs. If the context is not valid or has not been initialized, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure. Must not be null and should point to a valid, initialized context. If the context is invalid, the function's behavior is undefined.
- **Output**: Returns an `int64_t` representing the number of key-value pairs in the context. This value will be zero if there are no key-value pairs.
- **See also**: [`gguf_get_n_kv`](../src/gguf.cpp.driver.md#gguf_get_n_kv)  (Implementation)


---
### gguf\_find\_key<!-- {{#callable_declaration:gguf_find_key}} -->
Finds the index of a key in the context.
- **Description**: This function is used to locate the index of a specified key within a `gguf_context`. It should be called after initializing the context and populating it with key-value pairs. If the key is found, the function returns its index; otherwise, it returns -1. This allows users to check for the existence of a key and retrieve its position for further operations.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that must be initialized and populated with key-value pairs. Must not be null.
    - `key`: A pointer to a null-terminated string representing the key to search for. Must not be null.
- **Output**: Returns the index of the key if found; otherwise, returns -1.
- **See also**: [`gguf_find_key`](../src/gguf.cpp.driver.md#gguf_find_key)  (Implementation)


---
### gguf\_get\_key<!-- {{#callable_declaration:gguf_get_key}} -->
Retrieves the key associated with a specified key ID.
- **Description**: This function is used to obtain the key string from a key-value pair in a GGUF context. It should be called after ensuring that the `gguf_context` has been properly initialized and that the specified `key_id` is valid. The function will assert if the `key_id` is out of bounds, meaning it must be a non-negative integer less than the total number of key-value pairs in the context. This is crucial for avoiding runtime errors.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that must not be null and should be properly initialized before calling this function.
    - `key_id`: An integer representing the index of the key to retrieve. It must be non-negative and less than the total number of key-value pairs in the context, as determined by `gguf_get_n_kv`. If an invalid `key_id` is provided, the function will assert and terminate the program.
- **Output**: Returns a pointer to the key string associated with the specified `key_id`. The returned string is valid as long as the `gguf_context` remains in scope and is not modified.
- **See also**: [`gguf_get_key`](../src/gguf.cpp.driver.md#gguf_get_key)  (Implementation)


---
### gguf\_get\_kv\_type<!-- {{#callable_declaration:gguf_get_kv_type}} -->
Retrieves the type of a key-value pair.
- **Description**: This function is used to obtain the type of a key-value pair identified by `key_id` within a given `gguf_context`. It should be called after ensuring that the `key_id` is valid, specifically that it is non-negative and less than the total number of key-value pairs in the context. If an invalid `key_id` is provided, the behavior is undefined, so it is crucial to validate the `key_id` using the `gguf_get_n_kv` function before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds the key-value pairs. Must not be null.
    - `key_id`: An integer representing the index of the key-value pair. It must be in the range [0, n_kv), where n_kv is the total number of key-value pairs in the context. Providing an out-of-bounds value will lead to undefined behavior.
- **Output**: Returns the type of the key-value pair as an `enum gguf_type`, which indicates whether the value is a specific data type or an array.
- **See also**: [`gguf_get_kv_type`](../src/gguf.cpp.driver.md#gguf_get_kv_type)  (Implementation)


---
### gguf\_get\_arr\_type<!-- {{#callable_declaration:gguf_get_arr_type}} -->
Retrieves the type of an array stored in a GGUF context.
- **Description**: This function is used to obtain the type of an array associated with a specific key in a GGUF context. It should be called only after ensuring that the key corresponds to an array type; otherwise, it may lead to undefined behavior. The function requires a valid `gguf_context` pointer and a `key_id` that must be within the valid range of key-value pairs. If the provided `key_id` does not point to an array, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that must not be null and should be properly initialized. It represents the context from which the array type is being queried.
    - `key_id`: An integer representing the index of the key-value pair in the context. It must be a non-negative integer and less than the total number of key-value pairs in the context, as determined by `gguf_get_n_kv(ctx)`. If `key_id` is out of bounds, the behavior is undefined.
- **Output**: Returns an enumeration value of type `gguf_type` that indicates the type of the array. If the key does not correspond to an array, the behavior is undefined.
- **See also**: [`gguf_get_arr_type`](../src/gguf.cpp.driver.md#gguf_get_arr_type)  (Implementation)


---
### gguf\_get\_val\_u8<!-- {{#callable_declaration:gguf_get_val_u8}} -->
Retrieves a uint8_t value associated with a specific key.
- **Description**: This function is used to obtain a `uint8_t` value from a GGUF context based on a provided key identifier. It should be called after ensuring that the context is properly initialized and that the key identifier is valid. The function will assert if the key identifier is out of bounds or if the associated value type is not a single `uint8_t`. Therefore, it is crucial to verify that the key exists and that its type is correct before calling this function to avoid runtime errors.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the GGUF context. Must not be null and should be properly initialized before use.
    - `key_id`: An integer representing the key identifier for the desired value. It must be non-negative and less than the total number of key-value pairs in the context, as returned by `gguf_get_n_kv(ctx)`. If the key_id is invalid, the function will assert and terminate the program.
- **Output**: Returns the `uint8_t` value associated with the specified key identifier. If the key identifier does not correspond to a valid `uint8_t` value, the behavior is undefined.
- **See also**: [`gguf_get_val_u8`](../src/gguf.cpp.driver.md#gguf_get_val_u8)  (Implementation)


---
### gguf\_get\_val\_i8<!-- {{#callable_declaration:gguf_get_val_i8}} -->
Retrieves an 8-bit signed integer value from the context.
- **Description**: This function is used to obtain an 8-bit signed integer value associated with a specific key in the given context. It is essential to ensure that the `key_id` provided is valid, meaning it must be non-negative and less than the total number of key-value pairs in the context. Additionally, the key must correspond to a value that is of the correct type; specifically, it should have exactly one element. If these conditions are not met, the function will trigger an assertion failure. This function should be called after the context has been properly initialized and populated with key-value pairs.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure, which must not be null and should be properly initialized before calling this function.
    - `key_id`: An integer representing the index of the key-value pair to retrieve. It must be in the range [0, n_kv), where n_kv is the total number of key-value pairs in the context. If the key_id is out of this range, the function will trigger an assertion failure.
- **Output**: Returns the 8-bit signed integer value associated with the specified key_id in the context.
- **See also**: [`gguf_get_val_i8`](../src/gguf.cpp.driver.md#gguf_get_val_i8)  (Implementation)


---
### gguf\_get\_val\_u16<!-- {{#callable_declaration:gguf_get_val_u16}} -->
Retrieves a 16-bit unsigned integer value from the context.
- **Description**: This function is used to obtain a 16-bit unsigned integer value associated with a specific key in the given context. It should be called after ensuring that the key ID is valid and corresponds to a key-value pair that holds a value of the correct type. If the key ID is out of bounds or if the value type is not a 16-bit unsigned integer, the function will trigger an assertion failure. Therefore, it is essential to validate the key ID and type before calling this function to avoid runtime errors.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds the key-value pairs. Must not be null.
    - `key_id`: An integer representing the ID of the key whose value is to be retrieved. It must be non-negative and less than the total number of key-value pairs in the context.
- **Output**: Returns the 16-bit unsigned integer value associated with the specified key ID.
- **See also**: [`gguf_get_val_u16`](../src/gguf.cpp.driver.md#gguf_get_val_u16)  (Implementation)


---
### gguf\_get\_val\_i16<!-- {{#callable_declaration:gguf_get_val_i16}} -->
Retrieves a 16-bit signed integer value from the context.
- **Description**: This function is used to obtain a 16-bit signed integer value associated with a specific key in the given context. It should be called only after ensuring that the key exists and is of the correct type, as indicated by the context's key-value pairs. The function will assert if the provided `key_id` is out of bounds or if the value type for the specified key is not a single value. It is important to validate that the context is properly initialized before calling this function.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure, which must not be null and should be properly initialized. It represents the context from which the value is retrieved.
    - `key_id`: An integer representing the index of the key-value pair. It must be non-negative and less than the total number of key-value pairs in the context. If the `key_id` is invalid, the function will assert and terminate the program.
- **Output**: Returns the 16-bit signed integer value associated with the specified key. If the key does not correspond to a valid single value, the behavior is undefined.
- **See also**: [`gguf_get_val_i16`](../src/gguf.cpp.driver.md#gguf_get_val_i16)  (Implementation)


---
### gguf\_get\_val\_u32<!-- {{#callable_declaration:gguf_get_val_u32}} -->
Retrieves a 32-bit unsigned integer value from the context.
- **Description**: This function is used to obtain a 32-bit unsigned integer value associated with a specific key in the given context. It must be called with a valid `gguf_context` that has been properly initialized and contains key-value pairs. The `key_id` parameter must be a non-negative integer that is less than the total number of key-value pairs in the context. If the key associated with `key_id` does not hold a value of type `GGUF_TYPE_UINT32`, the function will result in an assertion failure. Therefore, it is essential to ensure that the correct type is being accessed.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that must be valid and initialized. Caller retains ownership.
    - `key_id`: An integer representing the index of the key-value pair to retrieve. It must be non-negative and less than the total number of key-value pairs in the context.
- **Output**: Returns the 32-bit unsigned integer value associated with the specified key. The behavior is undefined if the key does not exist or if the value type is not `GGUF_TYPE_UINT32`.
- **See also**: [`gguf_get_val_u32`](../src/gguf.cpp.driver.md#gguf_get_val_u32)  (Implementation)


---
### gguf\_get\_val\_i32<!-- {{#callable_declaration:gguf_get_val_i32}} -->
Retrieves a 32-bit integer value from the context using a specified key.
- **Description**: This function is used to obtain a 32-bit integer value associated with a specific key in the given context. It should be called after ensuring that the key is valid and corresponds to a value of the correct type. The function will assert if the key is out of bounds or if the associated value is not of type `int32_t`. It is important to ensure that the context has been properly initialized before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds the key-value pairs. Must not be null and should be properly initialized.
    - `key_id`: An integer representing the index of the key-value pair to retrieve. It must be non-negative and less than the total number of key-value pairs in the context, as returned by `gguf_get_n_kv(ctx)`. If the key_id is invalid, the function will assert.
- **Output**: Returns the 32-bit integer value associated with the specified key. If the key does not correspond to an `int32_t` value, the function will assert.
- **See also**: [`gguf_get_val_i32`](../src/gguf.cpp.driver.md#gguf_get_val_i32)  (Implementation)


---
### gguf\_get\_val\_f32<!-- {{#callable_declaration:gguf_get_val_f32}} -->
Retrieves a 32-bit floating-point value from the context.
- **Description**: This function is used to obtain a 32-bit floating-point value associated with a specific key in the context. It should be called after ensuring that the key ID is valid and corresponds to a value of the correct type (float). If the key ID is out of bounds or does not correspond to a float value, the function will assert and terminate the program. Therefore, it is essential to validate the key ID and its type before calling this function to avoid unexpected behavior.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds the key-value pairs. Must not be null.
    - `key_id`: An integer representing the key ID for the desired value. It must be non-negative and less than the total number of key-value pairs in the context.
- **Output**: Returns the 32-bit floating-point value associated with the specified key ID.
- **See also**: [`gguf_get_val_f32`](../src/gguf.cpp.driver.md#gguf_get_val_f32)  (Implementation)


---
### gguf\_get\_val\_u64<!-- {{#callable_declaration:gguf_get_val_u64}} -->
Retrieves a 64-bit unsigned integer value from the context.
- **Description**: This function is used to obtain a 64-bit unsigned integer value associated with a specific key in the given context. It must be called with a valid `key_id` that is non-negative and less than the total number of key-value pairs in the context. Additionally, the key associated with the provided `key_id` must have a value type of `GGUF_TYPE_UINT64`. If these conditions are not met, the function will trigger an assertion failure.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds the key-value pairs. Must not be null.
    - `key_id`: An integer representing the index of the key-value pair to retrieve. It must be non-negative and less than the total number of key-value pairs in the context.
- **Output**: Returns the 64-bit unsigned integer value associated with the specified key. If the key does not correspond to a `GGUF_TYPE_UINT64`, the behavior is undefined.
- **See also**: [`gguf_get_val_u64`](../src/gguf.cpp.driver.md#gguf_get_val_u64)  (Implementation)


---
### gguf\_get\_val\_i64<!-- {{#callable_declaration:gguf_get_val_i64}} -->
Retrieves a 64-bit integer value from the context.
- **Description**: This function is used to obtain a 64-bit integer value associated with a specific key in the context. It should be called after ensuring that the key ID is valid and corresponds to a key-value pair that contains a single value of type `int64_t`. If the key ID is out of bounds or does not point to a valid entry, the behavior is undefined. It is important to verify that the key ID is within the valid range and that the associated value type is correct before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds the key-value pairs. Must not be null.
    - `key_id`: An integer representing the ID of the key whose value is to be retrieved. Must be non-negative and less than the total number of key-value pairs in the context.
- **Output**: Returns the 64-bit integer value associated with the specified key ID.
- **See also**: [`gguf_get_val_i64`](../src/gguf.cpp.driver.md#gguf_get_val_i64)  (Implementation)


---
### gguf\_get\_val\_f64<!-- {{#callable_declaration:gguf_get_val_f64}} -->
Retrieves a 64-bit floating-point value from the context.
- **Description**: This function is used to obtain a 64-bit floating-point value associated with a specific key in the context. It must be called with a valid `gguf_context` that has been properly initialized and contains key-value pairs. The `key_id` parameter must be a valid index within the range of existing key-value pairs, and it is expected that the value associated with this key is of type `double`. If the `key_id` is out of bounds or if the value type does not match, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that must not be null and should be properly initialized. It represents the context from which the value is retrieved.
    - `key_id`: An integer representing the index of the key-value pair. It must be non-negative and less than the total number of key-value pairs in the context, as returned by `gguf_get_n_kv(ctx)`. If the index is invalid, the function will assert and terminate the program.
- **Output**: Returns the 64-bit floating-point value associated with the specified key. If the key does not correspond to a `double` type, the behavior is undefined.
- **See also**: [`gguf_get_val_f64`](../src/gguf.cpp.driver.md#gguf_get_val_f64)  (Implementation)


---
### gguf\_get\_val\_bool<!-- {{#callable_declaration:gguf_get_val_bool}} -->
Retrieves a boolean value from the context using a specified key.
- **Description**: This function is used to obtain a boolean value associated with a specific key in the given context. It should be called after ensuring that the key is valid and corresponds to a boolean type. The function will assert if the key is out of bounds or if the value type for the specified key is not a boolean. It is important to ensure that the context is properly initialized before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds the key-value pairs. Must not be null.
    - `key_id`: An integer representing the index of the key-value pair to retrieve. Must be non-negative and less than the total number of key-value pairs in the context.
- **Output**: Returns the boolean value associated with the specified key. If the key is invalid or does not correspond to a boolean type, the function will assert and terminate the program.
- **See also**: [`gguf_get_val_bool`](../src/gguf.cpp.driver.md#gguf_get_val_bool)  (Implementation)


---
### gguf\_get\_val\_str<!-- {{#callable_declaration:gguf_get_val_str}} -->
Retrieves the string value associated with a given key ID.
- **Description**: This function is used to obtain the string value from a key-value pair in a GGUF context. It should be called after ensuring that the `gguf_context` has been properly initialized and that the specified `key_id` is valid. The function will assert if the `key_id` is out of bounds or if the associated value is not a string. It is important to note that the returned string is valid only as long as the `gguf_context` remains unchanged.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure. Must not be null and must point to a valid context that has been initialized.
    - `key_id`: An integer representing the ID of the key-value pair. It must be non-negative and less than the total number of key-value pairs in the context. If the `key_id` is invalid, the function will assert.
- **Output**: Returns a pointer to a constant string representing the value associated with the specified key ID. The string is valid only while the `gguf_context` remains unchanged.
- **See also**: [`gguf_get_val_str`](../src/gguf.cpp.driver.md#gguf_get_val_str)  (Implementation)


---
### gguf\_get\_val\_data<!-- {{#callable_declaration:gguf_get_val_data}} -->
Retrieves the binary data associated with a specified key.
- **Description**: This function is used to obtain the raw binary data corresponding to a key-value pair identified by `key_id` within a `gguf_context`. It is essential to ensure that the `key_id` is valid, meaning it must be non-negative and less than the total number of key-value pairs in the context. Additionally, the value type associated with the specified key must not be a string, and it must have exactly one element. If these conditions are not met, the function will assert and terminate the program.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds the key-value pairs. Must not be null.
    - `key_id`: An integer identifier for the key-value pair. It must be in the range [0, number of key-value pairs - 1]. If invalid, the function will assert.
- **Output**: Returns a pointer to the binary data associated with the specified key. The data is valid only if the key's value type is not a string and has exactly one element.
- **See also**: [`gguf_get_val_data`](../src/gguf.cpp.driver.md#gguf_get_val_data)  (Implementation)


---
### gguf\_get\_arr\_n<!-- {{#callable_declaration:gguf_get_arr_n}} -->
Returns the number of elements in an array associated with a given key.
- **Description**: This function is used to retrieve the number of elements in an array stored in a GGUF context, identified by the specified key ID. It should be called after ensuring that the key ID is valid and corresponds to an array type. If the key ID is out of bounds or does not correspond to an array, the behavior is undefined. The function will assert if the key ID is invalid, ensuring that the caller is aware of the requirement for valid input.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the GGUF context. Must not be null.
    - `key_id`: An integer representing the key ID for which the number of array elements is requested. Must be non-negative and less than the total number of key-value pairs in the context.
- **Output**: Returns the number of elements in the array associated with the specified key ID. If the key ID corresponds to a string type, the return value will be the size of the string.
- **See also**: [`gguf_get_arr_n`](../src/gguf.cpp.driver.md#gguf_get_arr_n)  (Implementation)


---
### gguf\_get\_arr\_data<!-- {{#callable_declaration:gguf_get_arr_data}} -->
Retrieves a pointer to the data of an array stored in the context.
- **Description**: This function is used to access the raw data of an array associated with a specific key in the given context. It should be called after ensuring that the key corresponds to an array type, as indicated by the `gguf_get_kv_type` function. The `key_id` must be a valid index within the range of key-value pairs in the context, and it must not correspond to a string type. If the provided `key_id` is invalid or does not point to an array, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure, which must not be null and should be properly initialized before calling this function.
    - `key_id`: An integer representing the index of the key-value pair. It must be non-negative and less than the total number of key-value pairs in the context, as returned by `gguf_get_n_kv`. If `key_id` points to a non-array type, the behavior is undefined.
- **Output**: Returns a pointer to the first element of the array associated with the specified `key_id`. If the `key_id` is invalid or does not correspond to an array, the behavior is undefined.
- **See also**: [`gguf_get_arr_data`](../src/gguf.cpp.driver.md#gguf_get_arr_data)  (Implementation)


---
### gguf\_get\_arr\_str<!-- {{#callable_declaration:gguf_get_arr_str}} -->
Retrieves a C string from an array stored in a GGUF context.
- **Description**: This function is used to access a specific string from an array associated with a key in a GGUF context. It should be called after ensuring that the `key_id` corresponds to a key that holds an array of strings, and that the index `i` is within the bounds of the array. If the `key_id` does not point to a string array or if the index is out of range, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the GGUF context. Must not be null.
    - `key_id`: An integer identifier for the key in the GGUF context. It must be non-negative and less than the total number of key-value pairs in the context.
    - `i`: An index specifying which string to retrieve from the array. It must be a valid index within the bounds of the string array associated with the given `key_id`.
- **Output**: Returns a pointer to the C string at the specified index in the array. If the input parameters are valid, this string will be valid for use; otherwise, the behavior is undefined.
- **See also**: [`gguf_get_arr_str`](../src/gguf.cpp.driver.md#gguf_get_arr_str)  (Implementation)


---
### gguf\_get\_n\_tensors<!-- {{#callable_declaration:gguf_get_n_tensors}} -->
Returns the number of tensors in the context.
- **Description**: This function is used to retrieve the total count of tensors associated with a given `gguf_context`. It should be called after the context has been properly initialized and populated with tensor data. The function will return a non-negative integer representing the number of tensors. If the context is not valid or uninitialized, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the context from which to retrieve the tensor count. This pointer must not be null and should point to a valid, initialized context. Passing a null pointer or an uninitialized context may lead to undefined behavior.
- **Output**: Returns an `int64_t` value indicating the number of tensors in the context. A return value of zero indicates that there are no tensors present.
- **See also**: [`gguf_get_n_tensors`](../src/gguf.cpp.driver.md#gguf_get_n_tensors)  (Implementation)


---
### gguf\_find\_tensor<!-- {{#callable_declaration:gguf_find_tensor}} -->
Finds the ID of a tensor by its name.
- **Description**: This function is used to retrieve the ID of a tensor from a given context based on its name. It should be called after initializing the `gguf_context` and before attempting to access tensor data. If the specified tensor name does not exist within the context, the function will return -1, indicating that the tensor was not found.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the context from which to search for the tensor. Must not be null.
    - `name`: A string representing the name of the tensor to find. Must not be null and should be a valid C string.
- **Output**: Returns the ID of the tensor if found; otherwise, returns -1.
- **See also**: [`gguf_find_tensor`](../src/gguf.cpp.driver.md#gguf_find_tensor)  (Implementation)


---
### gguf\_get\_tensor\_offset<!-- {{#callable_declaration:gguf_get_tensor_offset}} -->
Retrieves the offset of a specified tensor.
- **Description**: This function is used to obtain the byte offset of a tensor within the binary data blob associated with a `gguf_context`. It should be called after ensuring that the `gguf_context` has been properly initialized and that the specified tensor ID is valid. The function will assert if the tensor ID is out of bounds, meaning it must be a non-negative integer less than the total number of tensors in the context.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that must not be null and should be properly initialized. It contains information about the tensors.
    - `tensor_id`: An integer representing the ID of the tensor whose offset is to be retrieved. It must be a non-negative integer less than the total number of tensors in the context.
- **Output**: Returns the byte offset of the specified tensor within the tensor data binary blob.
- **See also**: [`gguf_get_tensor_offset`](../src/gguf.cpp.driver.md#gguf_get_tensor_offset)  (Implementation)


---
### gguf\_get\_tensor\_name<!-- {{#callable_declaration:gguf_get_tensor_name}} -->
Retrieves the name of a tensor.
- **Description**: This function is used to obtain the name of a tensor identified by its `tensor_id` within a given `gguf_context`. It should be called after ensuring that the `tensor_id` is valid, specifically that it is non-negative and less than the total number of tensors in the context. If the `tensor_id` is out of bounds, the behavior is undefined, so it is crucial to validate the input before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds the context of the GGUF file. Must not be null.
    - `tensor_id`: An integer representing the index of the tensor whose name is to be retrieved. It must be a non-negative value and less than the total number of tensors in the context, as returned by `gguf_get_n_tensors`. Passing an invalid `tensor_id` will lead to undefined behavior.
- **Output**: Returns a pointer to a string containing the name of the specified tensor. The returned string is managed by the `gguf_context` and should not be modified or freed by the caller.
- **See also**: [`gguf_get_tensor_name`](../src/gguf.cpp.driver.md#gguf_get_tensor_name)  (Implementation)


---
### gguf\_get\_tensor\_type<!-- {{#callable_declaration:gguf_get_tensor_type}} -->
Retrieves the type of a specified tensor.
- **Description**: This function is used to obtain the data type of a tensor identified by its `tensor_id` within a given `gguf_context`. It must be called after initializing the context and ensures that the `tensor_id` is valid, meaning it should be non-negative and less than the total number of tensors in the context. If an invalid `tensor_id` is provided, the function will assert and terminate the program, so it is crucial to ensure that the `tensor_id` is within the valid range before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the context containing the tensor. Must not be null.
    - `tensor_id`: An integer identifier for the tensor whose type is to be retrieved. It must be a non-negative integer and less than the total number of tensors in the context.
- **Output**: Returns the `ggml_type` of the specified tensor, indicating the data type of the tensor.
- **See also**: [`gguf_get_tensor_type`](../src/gguf.cpp.driver.md#gguf_get_tensor_type)  (Implementation)


---
### gguf\_get\_tensor\_size<!-- {{#callable_declaration:gguf_get_tensor_size}} -->
Retrieves the size of a specified tensor.
- **Description**: This function is used to obtain the size in bytes of a tensor identified by its `tensor_id` within a given `gguf_context`. It should be called after ensuring that the `gguf_context` has been properly initialized and contains the specified tensor. The function will assert if the `tensor_id` is out of bounds, meaning it must be a non-negative integer and less than the total number of tensors in the context.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the context containing the tensor. Must not be null.
    - `tensor_id`: An integer identifier for the tensor whose size is to be retrieved. It must be a non-negative integer and less than the total number of tensors in the context.
- **Output**: Returns the size of the specified tensor in bytes.
- **See also**: [`gguf_get_tensor_size`](../src/gguf.cpp.driver.md#gguf_get_tensor_size)  (Implementation)


---
### gguf\_remove\_key<!-- {{#callable_declaration:gguf_remove_key}} -->
Removes a key from the context if it exists.
- **Description**: This function is used to remove a key-value pair from the `gguf_context`. It should be called when you want to delete a specific key from the context, which may be necessary for managing the data stored in the GGUF format. The function will return the identifier of the key that was removed, or -1 if the key was not found. It is important to ensure that the `ctx` parameter is valid and properly initialized before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the context from which the key should be removed. Must not be null and should be properly initialized.
    - `key`: A pointer to a null-terminated string representing the key to be removed. Must not be null. If the key does not exist in the context, the function will return -1.
- **Output**: Returns the identifier of the key that was removed from the context, or -1 if the key was not found.
- **See also**: [`gguf_remove_key`](../src/gguf.cpp.driver.md#gguf_remove_key)  (Implementation)


---
### gguf\_set\_val\_u8<!-- {{#callable_declaration:gguf_set_val_u8}} -->
Overrides or adds a key-value pair with an 8-bit unsigned integer value.
- **Description**: This function is used to set a key-value pair in the context, where the value is an 8-bit unsigned integer. It should be called when you want to store or update a value associated with a specific key in the `gguf_context`. The function first checks if the key is reserved and then removes any existing key-value pair associated with the same key before adding the new pair. It is important to ensure that the `ctx` parameter is properly initialized and not null before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the context in which the key-value pair is stored. Must not be null and must be properly initialized.
    - `key`: A string representing the key for the key-value pair. Must not be null and should not be a reserved key. If the key is invalid, the behavior is undefined.
    - `val`: An 8-bit unsigned integer value to be associated with the key. Valid values range from 0 to 255.
- **Output**: None
- **See also**: [`gguf_set_val_u8`](../src/gguf.cpp.driver.md#gguf_set_val_u8)  (Implementation)


---
### gguf\_set\_val\_i8<!-- {{#callable_declaration:gguf_set_val_i8}} -->
Sets a key-value pair with an 8-bit integer value.
- **Description**: This function is used to set or update a key-value pair in the context with an 8-bit integer value. It should be called after initializing the `gguf_context`. If the specified key already exists, it will be removed before adding the new key-value pair. The function does not perform any checks on the validity of the `val` parameter, but it is expected to be a valid 8-bit integer.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure, which must be initialized before calling this function. Must not be null.
    - `key`: A string representing the key for the key-value pair. Must not be null.
    - `val`: An 8-bit integer value to be associated with the specified key. Valid values are from -128 to 127.
- **Output**: None
- **See also**: [`gguf_set_val_i8`](../src/gguf.cpp.driver.md#gguf_set_val_i8)  (Implementation)


---
### gguf\_set\_val\_u16<!-- {{#callable_declaration:gguf_set_val_u16}} -->
Overrides or adds a key-value pair with a 16-bit unsigned integer value.
- **Description**: This function is used to set or update a key-value pair in the context with a 16-bit unsigned integer value. It should be called when you want to store a new value or replace an existing one associated with a specific key. The function first checks if the key is reserved and then removes any existing entry for that key before adding the new key-value pair. It is important to ensure that the `ctx` parameter is properly initialized and not null before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the current context. Must not be null and must be properly initialized.
    - `key`: A string representing the key for the key-value pair. Must not be null and should not be a reserved key. If the key is invalid, the function will handle it by checking against reserved keys.
    - `val`: A 16-bit unsigned integer value to be associated with the key. This value is stored in the context.
- **Output**: None
- **See also**: [`gguf_set_val_u16`](../src/gguf.cpp.driver.md#gguf_set_val_u16)  (Implementation)


---
### gguf\_set\_val\_i16<!-- {{#callable_declaration:gguf_set_val_i16}} -->
Sets a 16-bit integer value for a specified key.
- **Description**: This function is used to set or update a key-value pair in the context with a 16-bit integer value. It should be called when you want to store a new value or replace an existing one associated with the specified key. The function first checks if the key is reserved and then removes any existing key-value pair with the same key before adding the new pair. It is important to ensure that the `ctx` parameter is properly initialized and not null before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the context in which the key-value pair is stored. Must not be null.
    - `key`: A string representing the key for the key-value pair. Must not be null and should not be a reserved key.
    - `val`: The 16-bit integer value to be associated with the key. This value can be any valid `int16_t` value.
- **Output**: None
- **See also**: [`gguf_set_val_i16`](../src/gguf.cpp.driver.md#gguf_set_val_i16)  (Implementation)


---
### gguf\_set\_val\_u32<!-- {{#callable_declaration:gguf_set_val_u32}} -->
Overrides or adds a key-value pair with a 32-bit unsigned integer value.
- **Description**: This function is used to set or update a key-value pair in the context with a 32-bit unsigned integer. It should be called when you want to store a new value or replace an existing one associated with the specified key. The function first checks if the key is reserved and then removes any existing entry for that key before adding the new key-value pair. It is important to ensure that the `ctx` parameter is properly initialized and not null before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `struct gguf_context` that represents the context in which the key-value pair is stored. Must not be null.
    - `key`: A string representing the key for the key-value pair. Must not be null and should not be a reserved key.
    - `val`: A 32-bit unsigned integer value to be associated with the key. Valid values are in the range of 0 to 4294967295.
- **Output**: None
- **See also**: [`gguf_set_val_u32`](../src/gguf.cpp.driver.md#gguf_set_val_u32)  (Implementation)


---
### gguf\_set\_val\_i32<!-- {{#callable_declaration:gguf_set_val_i32}} -->
Sets a 32-bit integer value for a specified key.
- **Description**: This function is used to store or update a key-value pair in the context, where the value is a 32-bit integer. It should be called after initializing the `gguf_context`. If the specified key already exists, it will be removed before adding the new key-value pair. This ensures that the most recent value is always stored. It is important to ensure that the key is not a reserved key, as this may lead to undefined behavior.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure, which must be initialized and must not be null.
    - `key`: A string representing the key for the value. This must not be null and should not be a reserved key.
    - `val`: An integer value of type `int32_t` that will be associated with the specified key.
- **Output**: None
- **See also**: [`gguf_set_val_i32`](../src/gguf.cpp.driver.md#gguf_set_val_i32)  (Implementation)


---
### gguf\_set\_val\_f32<!-- {{#callable_declaration:gguf_set_val_f32}} -->
Sets a floating-point value in the key-value store.
- **Description**: This function is used to set or update a floating-point value associated with a specified key in the key-value store of a `gguf_context`. It is important to ensure that the `ctx` parameter is a valid context that has been properly initialized. The function will first check if the key is reserved and then remove any existing entry for that key before adding the new key-value pair. If the key is invalid or reserved, the function may trigger an error.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure. Must not be null and must be initialized before calling this function.
    - `key`: A string representing the key for the value. Must not be null and should not be a reserved key.
    - `val`: A floating-point value to be associated with the key. This value will be stored in the key-value store.
- **Output**: None
- **See also**: [`gguf_set_val_f32`](../src/gguf.cpp.driver.md#gguf_set_val_f32)  (Implementation)


---
### gguf\_set\_val\_u64<!-- {{#callable_declaration:gguf_set_val_u64}} -->
Overrides or adds a key-value pair with a 64-bit unsigned integer value.
- **Description**: This function is used to set or update a key-value pair in the context with a 64-bit unsigned integer. It should be called when you want to store a new value or replace an existing one associated with a specific key. The function first checks if the key is reserved and then removes any existing entry for that key before adding the new key-value pair. It is important to ensure that the `ctx` parameter is properly initialized before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the context in which the key-value pair is stored. Must not be null and must be initialized before use.
    - `key`: A string representing the key for the key-value pair. Must not be null and should not be a reserved key. The function will handle invalid keys by checking against reserved keys.
    - `val`: A 64-bit unsigned integer value to be associated with the specified key. This value will be stored in the context.
- **Output**: None
- **See also**: [`gguf_set_val_u64`](../src/gguf.cpp.driver.md#gguf_set_val_u64)  (Implementation)


---
### gguf\_set\_val\_i64<!-- {{#callable_declaration:gguf_set_val_i64}} -->
Sets a key-value pair with a 64-bit integer value.
- **Description**: This function is used to set or update a key-value pair in the context with a 64-bit integer value. It should be called after initializing the `gguf_context`. If the specified key already exists, it will be removed before adding the new key-value pair. The function does not perform any checks on the validity of the `key` parameter, so it is the caller's responsibility to ensure that the key is appropriate for use.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the current context. Must not be null.
    - `key`: A string representing the key for the key-value pair. Must not be null.
    - `val`: An integer value of type `int64_t` to be associated with the key. There are no specific constraints on the value.
- **Output**: None
- **See also**: [`gguf_set_val_i64`](../src/gguf.cpp.driver.md#gguf_set_val_i64)  (Implementation)


---
### gguf\_set\_val\_f64<!-- {{#callable_declaration:gguf_set_val_f64}} -->
Sets a key-value pair with a double value.
- **Description**: This function is used to set or update a key-value pair in the context with a double value. It should be called when you want to store a new value or replace an existing one associated with the specified key. The function will first check if the key is reserved and then remove any existing entry for that key before adding the new key-value pair. It is important to ensure that the `ctx` parameter is properly initialized before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the current context. Must not be null and must be initialized before use.
    - `key`: A string representing the key for the key-value pair. Must not be null and should not be a reserved key. If the key is invalid, the function will handle it by checking against reserved keys.
    - `val`: A double value to be associated with the specified key. This value will be stored in the context.
- **Output**: None
- **See also**: [`gguf_set_val_f64`](../src/gguf.cpp.driver.md#gguf_set_val_f64)  (Implementation)


---
### gguf\_set\_val\_bool<!-- {{#callable_declaration:gguf_set_val_bool}} -->
Sets a boolean value in the key-value store.
- **Description**: This function is used to set a boolean value associated with a specified key in the key-value store of a `gguf_context`. It can be called at any time after the context has been initialized. If the key already exists, its previous value will be overridden. The function will check if the key is reserved and remove any existing entry for the key before adding the new key-value pair. It is important to ensure that the `ctx` parameter is valid and properly initialized before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that must be initialized and valid. Must not be null.
    - `key`: A string representing the key for the key-value pair. Must not be null and should not be a reserved key.
    - `val`: A boolean value to be associated with the key. Valid values are true or false.
- **Output**: None
- **See also**: [`gguf_set_val_bool`](../src/gguf.cpp.driver.md#gguf_set_val_bool)  (Implementation)


---
### gguf\_set\_val\_str<!-- {{#callable_declaration:gguf_set_val_str}} -->
Sets a string value for a specified key.
- **Description**: This function is used to set or update a key-value pair in the context, where the value is a string. It should be called when you want to store a new string value associated with a specific key, or to overwrite an existing value for that key. The function will first check if the key is reserved and will remove any existing entry for that key before adding the new key-value pair. It is important to ensure that the `ctx` parameter is properly initialized before calling this function.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that represents the current context. Must not be null and must be initialized before use.
    - `key`: A pointer to a null-terminated string that represents the key for the value being set. Must not be null and should not be a reserved key.
    - `val`: A pointer to a null-terminated string that represents the value to be associated with the key. Must not be null.
- **Output**: None
- **See also**: [`gguf_set_val_str`](../src/gguf.cpp.driver.md#gguf_set_val_str)  (Implementation)


---
### gguf\_set\_arr\_data<!-- {{#callable_declaration:gguf_set_arr_data}} -->
Creates a new array with specified data.
- **Description**: This function is used to create or update a key-value pair in the context, where the value is an array of a specified type. It should be called when you need to store an array of data associated with a key in the `gguf_context`. The function will first check for reserved keys and remove any existing key with the same name before adding the new array. It is important to ensure that the `ctx` is properly initialized before calling this function.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure where the array will be stored. Must not be null.
    - `key`: A string representing the key under which the array will be stored. Must not be null.
    - `type`: An enumeration value of type `gguf_type` indicating the data type of the array elements. Valid values are defined in the `gguf_type` enum.
    - `data`: A pointer to the data to be copied into the array. Must not be null.
    - `n`: The number of elements in the array. Must be greater than zero.
- **Output**: None
- **See also**: [`gguf_set_arr_data`](../src/gguf.cpp.driver.md#gguf_set_arr_data)  (Implementation)


---
### gguf\_set\_arr\_str<!-- {{#callable_declaration:gguf_set_arr_str}} -->
Creates or updates an array of strings in the context.
- **Description**: This function is used to set or update a key-value pair in the context, where the value is an array of strings. It should be called when you want to store multiple strings associated with a specific key. The function first checks for reserved keys and removes any existing entry for the specified key before adding the new array. It is important to ensure that the `ctx` parameter is properly initialized and that `data` contains valid string pointers. If `n` is zero, the function will still execute without error, effectively removing any existing array for the key.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that must be initialized before calling this function. Must not be null.
    - `key`: A string representing the key under which the array of strings will be stored. Must not be null.
    - `data`: An array of string pointers (const char**) that contains the strings to be stored. Must not be null. Each string must be a valid C string.
    - `n`: The number of strings in the `data` array. Must be greater than or equal to 0.
- **Output**: None
- **See also**: [`gguf_set_arr_str`](../src/gguf.cpp.driver.md#gguf_set_arr_str)  (Implementation)


---
### gguf\_set\_kv<!-- {{#callable_declaration:gguf_set_kv}} -->
Sets key-value pairs from one context to another.
- **Description**: This function is used to copy key-value pairs from a source `gguf_context` to a destination `gguf_context`. It should be called when you want to transfer data between contexts, ensuring that the destination context is properly initialized. The function handles various data types, including integers, floats, booleans, and strings, and it will ignore any key-value pairs that are arrays. If the source context contains invalid data types, the function will abort execution.
- **Inputs**:
    - `ctx`: A pointer to the destination `gguf_context` where key-value pairs will be set. Must not be null and should be properly initialized.
    - `src`: A pointer to the source `gguf_context` from which key-value pairs will be copied. Must not be null and should contain valid key-value pairs.
- **Output**: None
- **See also**: [`gguf_set_kv`](../src/gguf.cpp.driver.md#gguf_set_kv)  (Implementation)


---
### gguf\_add\_tensor<!-- {{#callable_declaration:gguf_add_tensor}} -->
Adds a tensor to the GGUF context.
- **Description**: This function is used to add a tensor to a GGUF context. It must be called after initializing the context and before writing the context to a file. The tensor's name must be unique within the context; if a tensor with the same name already exists, the function will abort. This function also calculates the offset for the new tensor based on the existing tensors in the context.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure where the tensor will be added. Must not be null.
    - `tensor`: A pointer to the `ggml_tensor` structure representing the tensor to be added. Must not be null.
- **Output**: None
- **See also**: [`gguf_add_tensor`](../src/gguf.cpp.driver.md#gguf_add_tensor)  (Implementation)


---
### gguf\_set\_tensor\_type<!-- {{#callable_declaration:gguf_set_tensor_type}} -->
Sets the type of a specified tensor.
- **Description**: This function is used to change the data type of a tensor identified by its name within a given context. It must be called after the context has been properly initialized and the tensor must already exist; otherwise, the function will abort. The function also recalculates the offsets of all tensors that follow the modified tensor to ensure that the tensor data remains contiguous in memory. It is important to ensure that the new type is compatible with the tensor's dimensions.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure that holds the tensor information. Must not be null.
    - `name`: A string representing the name of the tensor whose type is to be set. Must not be null and must correspond to an existing tensor in the context.
    - `type`: An enumeration value of type `ggml_type` that specifies the new type for the tensor. The value must be a valid type defined in the `ggml_type` enum.
- **Output**: None
- **See also**: [`gguf_set_tensor_type`](../src/gguf.cpp.driver.md#gguf_set_tensor_type)  (Implementation)


---
### gguf\_set\_tensor\_data<!-- {{#callable_declaration:gguf_set_tensor_data}} -->
Sets the data for a specified tensor.
- **Description**: This function is used to assign data to a tensor identified by its name within a given context. It must be called after the tensor has been added to the context and the tensor's name must be unique. If the specified tensor name does not exist, the function will abort execution. It is important to ensure that the data being set is compatible with the tensor's expected data type and size.
- **Inputs**:
    - `ctx`: A pointer to the `gguf_context` structure that holds the context for the GGUF file. Must not be null.
    - `name`: A string representing the name of the tensor to which the data will be assigned. Must not be null and must correspond to an existing tensor in the context.
    - `data`: A pointer to the data that will be assigned to the tensor. The data must be valid for the expected size of the tensor. Ownership of the data is not transferred; the caller is responsible for managing the memory.
- **Output**: None
- **See also**: [`gguf_set_tensor_data`](../src/gguf.cpp.driver.md#gguf_set_tensor_data)  (Implementation)


---
### gguf\_write\_to\_file<!-- {{#callable_declaration:gguf_write_to_file}} -->
Writes the GGUF context to a binary file.
- **Description**: This function is used to write the entire `gguf_context` to a specified binary file. It can be called when you want to persist the context data to disk, either fully or partially, depending on the `only_meta` parameter. If `only_meta` is set to true, only the metadata will be written, which can be useful for scenarios where tensor data is appended later. The function requires that the `ctx` parameter is a valid pointer to an initialized `gguf_context`, and the `fname` parameter must point to a valid file path where the data will be written. If the file cannot be opened for writing, the function will log an error and return false. It is important to ensure that the file path is accessible and writable.
- **Inputs**:
    - `ctx`: A pointer to a valid `gguf_context` that contains the data to be written. Must not be null.
    - `fname`: A string representing the file name (path) where the data will be written. Must not be null and should point to a valid writable location.
    - `only_meta`: A boolean flag indicating whether to write only the metadata (true) or the entire context including tensor data (false).
- **Output**: Returns true if the data was successfully written to the file; otherwise, returns false.
- **See also**: [`gguf_write_to_file`](../src/gguf.cpp.driver.md#gguf_write_to_file)  (Implementation)


---
### gguf\_get\_meta\_size<!-- {{#callable_declaration:gguf_get_meta_size}} -->
Retrieves the size of the metadata in bytes.
- **Description**: This function is used to obtain the size of the metadata associated with a `gguf_context`. It should be called after the context has been properly initialized. The size returned includes all relevant metadata such as the header, key-value pairs, and tensor information. It is important to ensure that the `ctx` parameter is valid and points to an initialized `gguf_context`; otherwise, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that holds the metadata. Must not be null and must point to a valid, initialized context.
- **Output**: Returns the size of the metadata in bytes as a `size_t` value.
- **See also**: [`gguf_get_meta_size`](../src/gguf.cpp.driver.md#gguf_get_meta_size)  (Implementation)


---
### gguf\_get\_meta\_data<!-- {{#callable_declaration:gguf_get_meta_data}} -->
Writes the meta data to the specified memory location.
- **Description**: This function is used to retrieve the meta data from a `gguf_context` and write it to a provided memory location. It should be called after the `gguf_context` has been properly initialized and populated with data. The caller is responsible for ensuring that the `data` buffer is large enough to hold the meta data, which can be determined using the `gguf_get_meta_size` function. If the provided `ctx` is null, the behavior is undefined.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure that contains the meta data to be written. Must not be null and must be initialized before calling this function.
    - `data`: A pointer to a memory location where the meta data will be written. The caller must ensure that this buffer is allocated and large enough to hold the meta data.
- **Output**: None
- **See also**: [`gguf_get_meta_data`](../src/gguf.cpp.driver.md#gguf_get_meta_data)  (Implementation)


