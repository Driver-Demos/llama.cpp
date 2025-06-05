# Purpose
The provided C++ source code file is a comprehensive implementation of a model loader for the "Llama" framework, which is designed to handle the loading and management of model data stored in GGUF (General Graph Universal Format) files. The file includes functionality for reading model metadata, managing tensor data, and handling various data types and formats. It defines a class [`llama_model_loader`](#llama_model_loaderllama_model_loader) that encapsulates the logic for loading model files, managing tensor data, and handling key-value metadata associated with the model. The code is structured to support both memory-mapped file access and direct file reading, allowing for efficient data handling depending on the platform capabilities.

Key components of the code include functions for determining file versions and model data types, managing file splits for large models, and handling metadata through a templated structure `GKV_Base` that supports various data types. The code also includes mechanisms for validating tensor data, managing memory mappings, and providing detailed logging and error handling. The [`llama_model_loader`](#llama_model_loaderllama_model_loader) class provides a public API for loading model data, accessing tensor metadata, and retrieving specific model parameters, making it a critical component for applications that need to load and utilize Llama models efficiently. The file is intended to be part of a larger system, likely imported and used by other components that require model loading capabilities.
# Imports and Dependencies

---
- `llama-model-loader.h`
- `ggml.h`
- `array`
- `cinttypes`
- `cstring`
- `future`


# Global Variables

---
### kiB
- **Type**: `size_t`
- **Description**: The variable `kiB` is a constant of type `size_t` representing the number of bytes in a kilobyte, specifically set to 1024 bytes. This is a common definition used in computing to denote a kilobyte in binary terms, as opposed to the decimal definition of 1000 bytes.
- **Use**: `kiB` is used as a base unit to define larger memory sizes, such as `MiB` and `GiB`, in the code.


---
### MiB
- **Type**: `size_t`
- **Description**: The `MiB` variable is a constant of type `size_t` that represents the number of bytes in a mebibyte. It is defined as 1024 times the number of bytes in a kibibyte (`kiB`), which is 1024 bytes. This makes `MiB` equal to 1,048,576 bytes.
- **Use**: `MiB` is used to represent memory sizes in mebibytes, facilitating calculations and conversions involving memory units.


---
### GiB
- **Type**: `size_t`
- **Description**: The variable `GiB` is a constant of type `size_t` that represents the number of bytes in a gibibyte. It is calculated as 1024 times the value of `MiB`, which itself is 1024 times the value of `kiB`, where `kiB` is 1024 bytes.
- **Use**: This variable is used to define a standard size for a gibibyte in bytes, which can be used for memory size calculations or comparisons in the program.


# Data Structures

---
### GKV\_Base\_Type<!-- {{#data_structure:GGUFMeta::GKV_Base_Type}} -->
- **Type**: `struct`
- **Members**:
    - `gt`: A static constant representing the gguf_type associated with the template specialization.
- **Description**: The `GKV_Base_Type` is a templated struct designed to provide a base type for key-value operations in the GGUF metadata context. It is parameterized by a type `T`, a `gguf_type` `gt_`, and a function pointer `gfun` that retrieves a value of type `T` from a `gguf_context` given an integer key. The struct defines a static constant `gt` to hold the `gguf_type` and a static method `getter` that uses the function pointer `gfun` to fetch the value from the context.
- **Member Functions**:
    - [`GGUFMeta::GKV_Base_Type::getter`](#GKV_Base_Typegetter)

**Methods**

---
#### GKV\_Base\_Type::getter<!-- {{#callable:GGUFMeta::GKV_Base_Type::getter}} -->
The `getter` function retrieves a value of type `T` from a `gguf_context` using a specified key identifier by invoking a provided function pointer `gfun`.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure from which the value is to be retrieved.
    - `kid`: An integer key identifier used to specify which value to retrieve from the context.
- **Control Flow**:
    - The function directly calls the function pointer `gfun` with `ctx` and `kid` as arguments.
    - The result of `gfun` is returned as the output of the `getter` function.
- **Output**: The function returns a value of type `T`, which is the result of the `gfun` function call.
- **See also**: [`GGUFMeta::GKV_Base_Type`](#GGUFMetaGKV_Base_Type)  (Data Structure)



---
### ArrayInfo<!-- {{#data_structure:GGUFMeta::ArrayInfo}} -->
- **Type**: `struct`
- **Members**:
    - `gt`: Represents the type of the array using the `gguf_type` enumeration.
    - `length`: Stores the number of elements in the array.
    - `data`: Points to the data of the array, which is of a generic type `void *`.
- **Description**: The `ArrayInfo` struct is designed to encapsulate metadata about an array, specifically within the context of the GGUF (Generic Graphical User Format) system. It holds information about the type of the array (`gt`), the number of elements it contains (`length`), and a pointer to the actual data (`data`). This structure is useful for managing arrays of various types and sizes, providing a uniform way to access array metadata and data in a type-agnostic manner.


---
### GKV<!-- {{#data_structure:GGUFMeta::GKV}} -->
- **Type**: `class`
- **Description**: The `GKV` class is a template class that extends `GKV_Base` and provides functionality for retrieving and setting key-value pairs from a context, with support for type validation and metadata overrides. It includes static methods for getting key-value pairs, converting override types to strings, validating overrides, and setting values with optional overrides. The class is designed to handle different data types, including boolean, integer, floating-point, and string types, and ensures that the retrieved or overridden values match the expected types.
- **Member Functions**:
    - [`GGUFMeta::GKV::GKV`](#GKVGKV)
    - [`GGUFMeta::GKV::get_kv`](#GKVget_kv)
    - [`GGUFMeta::GKV::override_type_to_str`](#GKVoverride_type_to_str)
    - [`GGUFMeta::GKV::validate_override`](#GKVvalidate_override)
    - [`GGUFMeta::GKV::try_override`](#GKVtry_override)
    - [`GGUFMeta::GKV::try_override`](#GKVtry_override)
    - [`GGUFMeta::GKV::try_override`](#GKVtry_override)
    - [`GGUFMeta::GKV::try_override`](#GKVtry_override)
    - [`GGUFMeta::GKV::set`](#GKVset)
    - [`GGUFMeta::GKV::set`](#GKVset)
    - [`GGUFMeta::GKV::set`](#GKVset)
- **Inherits From**:
    - `GKV_Base`

**Methods**

---
#### GKV::GKV<!-- {{#callable:GGUFMeta::GKV::GKV}} -->
The constructor `GKV()` is deleted to prevent instantiation of the `GKV` class.
- **Inputs**: None
- **Control Flow**:
    - The constructor `GKV()` is explicitly deleted, which means that any attempt to instantiate an object of the `GKV` class will result in a compilation error.
- **Output**: There is no output as the constructor is deleted and cannot be used.
- **See also**: [`GGUFMeta::GKV`](#GGUFMetaGKV)  (Data Structure)


---
#### GKV::get\_kv<!-- {{#callable:GGUFMeta::GKV::get_kv}} -->
The `get_kv` function retrieves a key-value pair from a context, ensuring the type matches the expected type for the template specialization.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure, which holds the context from which the key-value pair is retrieved.
    - `k`: An integer representing the key index within the context.
- **Control Flow**:
    - Retrieve the type of the key-value pair at index `k` using [`gguf_get_kv_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_kv_type) function.
    - Check if the retrieved type matches the expected type `GKV::gt` for the template specialization.
    - If the types do not match, throw a `std::runtime_error` with a formatted error message indicating the mismatch.
    - If the types match, return the value using the `GKV::getter` function.
- **Output**: Returns the value of the key-value pair at index `k` from the context, with the type determined by the template specialization.
- **Functions called**:
    - [`gguf_get_kv_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_kv_type)
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`gguf_get_key`](../ggml/src/gguf.cpp.driver.md#gguf_get_key)
    - [`gguf_type_name`](../ggml/src/gguf.cpp.driver.md#gguf_type_name)
- **See also**: [`GGUFMeta::GKV`](#GGUFMetaGKV)  (Data Structure)


---
#### GKV::override\_type\_to\_str<!-- {{#callable:GGUFMeta::GKV::override_type_to_str}} -->
The `override_type_to_str` function converts a `llama_model_kv_override_type` enumeration value to its corresponding string representation.
- **Inputs**:
    - `ty`: An enumeration value of type `llama_model_kv_override_type` representing the override type.
- **Control Flow**:
    - The function uses a switch statement to match the input enumeration value `ty` against predefined cases.
    - For each case, it returns a string representing the type: 'bool', 'int', 'float', or 'str'.
    - If the input does not match any predefined case, it returns 'unknown'.
- **Output**: A constant character pointer to a string representing the type name corresponding to the input enumeration value.
- **See also**: [`GGUFMeta::GKV`](#GGUFMetaGKV)  (Data Structure)


---
#### GKV::validate\_override<!-- {{#callable:GGUFMeta::GKV::validate_override}} -->
The `validate_override` function checks if a given metadata override matches an expected type and logs the override details if it does.
- **Inputs**:
    - `expected_type`: The expected type of the metadata override, specified as a `llama_model_kv_override_type` enum value.
    - `ovrd`: A pointer to a `llama_model_kv_override` structure that contains the metadata override to be validated.
- **Control Flow**:
    - Check if the `ovrd` pointer is null; if so, return false.
    - Compare the `tag` of the `ovrd` with the `expected_type`; if they match, log the override details based on the type and return true.
    - If the `tag` does not match the `expected_type`, log a warning message and return false.
- **Output**: Returns a boolean value: true if the override type matches the expected type and false otherwise.
- **Functions called**:
    - [`GGUFMeta::GKV::override_type_to_str`](#GKVoverride_type_to_str)
    - [`format`](llama-impl.cpp.driver.md#format)
- **See also**: [`GGUFMeta::GKV`](#GGUFMetaGKV)  (Data Structure)


---
#### GKV::try\_override<!-- {{#callable:GGUFMeta::GKV::try_override}} -->
The `try_override` function attempts to override a boolean target variable with a value from a `llama_model_kv_override` structure if the override is valid.
- **Inputs**:
    - `target`: A reference to a boolean variable that will be overridden if the override is valid.
    - `ovrd`: A pointer to a `llama_model_kv_override` structure containing the override value and type information.
- **Control Flow**:
    - The function first calls [`validate_override`](#GKVvalidate_override) with `LLAMA_KV_OVERRIDE_TYPE_BOOL` and the `ovrd` pointer to check if the override is valid for a boolean type.
    - If [`validate_override`](#GKVvalidate_override) returns true, the `target` is set to the boolean value from `ovrd->val_bool`.
    - The function returns true if the override was successful, otherwise it returns false.
- **Output**: A boolean value indicating whether the override was successful (true) or not (false).
- **Functions called**:
    - [`GGUFMeta::GKV::validate_override`](#GKVvalidate_override)
- **See also**: [`GGUFMeta::GKV`](#GGUFMetaGKV)  (Data Structure)


---
#### GKV::try\_override<!-- {{#callable:GGUFMeta::GKV::try_override}} -->
The `try_override` function attempts to override an integral target variable with a value from a `llama_model_kv_override` structure if the override is valid and of the correct type.
- **Inputs**:
    - `target`: A reference to an integral type variable that is intended to be overridden.
    - `ovrd`: A pointer to a `llama_model_kv_override` structure containing the override value and type information.
- **Control Flow**:
    - Check if the override is valid and of type `LLAMA_KV_OVERRIDE_TYPE_INT` using [`validate_override`](#GKVvalidate_override) function.
    - If valid, set the `target` to the integer value from `ovrd->val_i64`.
    - Return `true` if the override was successful.
    - Return `false` if the override was not valid or not of the correct type.
- **Output**: Returns a boolean indicating whether the override was successful.
- **Functions called**:
    - [`GGUFMeta::GKV::validate_override`](#GKVvalidate_override)
- **See also**: [`GGUFMeta::GKV`](#GGUFMetaGKV)  (Data Structure)


---
#### GKV::try\_override<!-- {{#callable:GGUFMeta::GKV::try_override}} -->
The `try_override` function attempts to override a target variable of a floating-point type with a value from a given override structure if the override is valid.
- **Inputs**:
    - `target`: A reference to a variable of type `T` that is intended to be overridden.
    - `ovrd`: A pointer to a `llama_model_kv_override` structure that contains the override value and type information.
- **Control Flow**:
    - Check if the override is valid for a floating-point type using [`validate_override`](#GKVvalidate_override) with `LLAMA_KV_OVERRIDE_TYPE_FLOAT` and the provided override structure.
    - If the override is valid, set the `target` to the floating-point value `val_f64` from the override structure and return `true`.
    - If the override is not valid, return `false`.
- **Output**: Returns a boolean value indicating whether the override was successfully applied (true) or not (false).
- **Functions called**:
    - [`GGUFMeta::GKV::validate_override`](#GKVvalidate_override)
- **See also**: [`GGUFMeta::GKV`](#GGUFMetaGKV)  (Data Structure)


---
#### GKV::try\_override<!-- {{#callable:GGUFMeta::GKV::try_override}} -->
The `try_override` function attempts to override a target variable of type `std::string` with a value from a `llama_model_kv_override` structure if the override is valid.
- **Inputs**:
    - `target`: A reference to a variable of type `std::string` that will be overridden if the override is valid.
    - `ovrd`: A pointer to a `llama_model_kv_override` structure containing the override information.
- **Control Flow**:
    - The function first checks if the override is valid by calling [`validate_override`](#GKVvalidate_override) with `LLAMA_KV_OVERRIDE_TYPE_STR` and the `ovrd` pointer.
    - If the override is valid, the `target` is set to the string value from `ovrd->val_str`.
    - The function returns `true` if the override is successful, otherwise it returns `false`.
- **Output**: A boolean value indicating whether the override was successful (`true`) or not (`false`).
- **Functions called**:
    - [`GGUFMeta::GKV::validate_override`](#GKVvalidate_override)
- **See also**: [`GGUFMeta::GKV`](#GGUFMetaGKV)  (Data Structure)


---
#### GKV::set<!-- {{#callable:GGUFMeta::GKV::set}} -->
The `set` function attempts to set a target variable to a value from a key-value context, with an optional override.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` object, which provides the context for key-value retrieval.
    - `k`: An integer representing the key index in the context from which to retrieve the value.
    - `target`: A reference to a variable of type `T` that will be set to the retrieved value.
    - `ovrd`: An optional pointer to a `llama_model_kv_override` structure, which may provide an override value for the target.
- **Control Flow**:
    - The function first attempts to override the target using the `try_override` function with the provided override structure `ovrd`.
    - If the override is successful, the function returns `true`.
    - If the key `k` is less than 0, the function returns `false` as it indicates an invalid key.
    - If no override is applied and the key is valid, the function retrieves the value associated with the key `k` from the context `ctx` using [`get_kv`](#GKVget_kv) and assigns it to `target`.
    - Finally, the function returns `true` to indicate successful setting of the target.
- **Output**: A boolean value indicating whether the target was successfully set, either through an override or by retrieving a value from the context.
- **Functions called**:
    - [`GGUFMeta::GKV::get_kv`](#GKVget_kv)
- **See also**: [`GGUFMeta::GKV`](#GGUFMetaGKV)  (Data Structure)


---
#### GKV::set<!-- {{#callable:GGUFMeta::GKV::set}} -->
The [`set`](#GKVset) function assigns a value to a target variable based on a key from a context, with an optional override.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` object, which provides the context for key-value operations.
    - `key`: A C-style string representing the key whose value is to be set in the target.
    - `target`: A reference to a variable of type `T` where the value associated with the key will be stored.
    - `ovrd`: An optional pointer to a `llama_model_kv_override` structure, which can override the value associated with the key.
- **Control Flow**:
    - The function calls another overloaded [`set`](#GKVset) function, passing the context, the result of `gguf_find_key(ctx, key)`, the target, and the override.
    - The [`gguf_find_key`](../ggml/src/gguf.cpp.driver.md#gguf_find_key) function is used to find the integer key corresponding to the string key in the context.
- **Output**: Returns a boolean indicating whether the value was successfully set in the target.
- **Functions called**:
    - [`GGUFMeta::GKV::set`](#GKVset)
    - [`gguf_find_key`](../ggml/src/gguf.cpp.driver.md#gguf_find_key)
- **See also**: [`GGUFMeta::GKV`](#GGUFMetaGKV)  (Data Structure)


---
#### GKV::set<!-- {{#callable:GGUFMeta::GKV::set}} -->
The [`set`](#GKVset) function assigns a value to a target variable based on a key from a context, with an optional override.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` object, which provides the context for key-value operations.
    - `key`: A `std::string` representing the key used to find the corresponding value in the context.
    - `target`: A reference to a variable of type `T` where the value will be stored.
    - `ovrd`: An optional pointer to a `llama_model_kv_override` structure, which can override the value from the context.
- **Control Flow**:
    - The function calls another overloaded [`set`](#GKVset) function, converting the `std::string` key to a C-style string using `c_str()`.
    - The actual logic of setting the value is handled by the [`set`](#GKVset) function that takes a `const char*` key.
- **Output**: Returns a boolean indicating whether the value was successfully set.
- **Functions called**:
    - [`GGUFMeta::GKV::set`](#GKVset)
- **See also**: [`GGUFMeta::GKV`](#GGUFMetaGKV)  (Data Structure)



---
### llama\_model\_loader<!-- {{#data_structure:llama_model_loader}} -->
- **Description**: [See definition](llama-model-loader.h.driver.md#llama_model_loader)
- **Member Functions**:
    - [`llama_model_loader::get_arr_n`](#llama_model_loaderget_arr_n)
    - [`llama_model_loader::get_arr_n`](#llama_model_loaderget_arr_n)
    - [`llama_model_loader::get_arr`](#llama_model_loaderget_arr)
    - [`llama_model_loader::get_arr`](#llama_model_loaderget_arr)
    - [`llama_model_loader::get_arr`](#llama_model_loaderget_arr)
    - [`llama_model_loader::get_key`](#llama_model_loaderget_key)
    - [`llama_model_loader::get_key`](#llama_model_loaderget_key)
    - [`llama_model_loader::get_key`](#llama_model_loaderget_key)
    - [`llama_model_loader::get_key_or_arr`](#llama_model_loaderget_key_or_arr)
    - [`llama_model_loader::get_key_or_arr`](#llama_model_loaderget_key_or_arr)
    - [`llama_model_loader::llama_model_loader`](#llama_model_loaderllama_model_loader)
    - [`llama_model_loader::get_arch_name`](#llama_model_loaderget_arch_name)
    - [`llama_model_loader::get_arch`](#llama_model_loaderget_arch)
    - [`llama_model_loader::get_weight`](#llama_model_loaderget_weight)
    - [`llama_model_loader::require_weight`](#llama_model_loaderrequire_weight)
    - [`llama_model_loader::get_tensor_meta`](#llama_model_loaderget_tensor_meta)
    - [`llama_model_loader::require_tensor_meta`](#llama_model_loaderrequire_tensor_meta)
    - [`llama_model_loader::check_tensor_dims`](#llama_model_loadercheck_tensor_dims)
    - [`llama_model_loader::create_tensor`](#llama_model_loadercreate_tensor)
    - [`llama_model_loader::create_tensor_as_view`](#llama_model_loadercreate_tensor_as_view)
    - [`llama_model_loader::done_getting_tensors`](#llama_model_loaderdone_getting_tensors)
    - [`llama_model_loader::init_mappings`](#llama_model_loaderinit_mappings)
    - [`llama_model_loader::get_mapping_range`](#llama_model_loaderget_mapping_range)
    - [`llama_model_loader::load_data_for`](#llama_model_loaderload_data_for)
    - [`llama_model_loader::load_all_data`](#llama_model_loaderload_all_data)
    - [`llama_model_loader::ftype_name`](#llama_model_loaderftype_name)
    - [`llama_model_loader::print_info`](#llama_model_loaderprint_info)

**Methods**

---
#### llama\_model\_loader::get\_arr\_n<!-- {{#callable:llama_model_loader::get_arr_n}} -->
The `get_arr_n` function retrieves the length of an array associated with a given key from a metadata context and stores it in the provided result variable.
- **Inputs**:
    - `key`: A string representing the key to search for in the metadata context.
    - `result`: A reference to an integral type variable where the length of the array will be stored.
    - `required`: A boolean indicating whether the key is required to be found; if true and the key is not found, an exception is thrown.
- **Control Flow**:
    - The function first attempts to find the key in the metadata context using [`gguf_find_key`](../ggml/src/gguf.cpp.driver.md#gguf_find_key).
    - If the key is not found and `required` is true, it throws a runtime error; otherwise, it returns false.
    - If the key is found, it retrieves the array information using `GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv`.
    - The length of the array is stored in the `result` variable.
    - The function returns true to indicate success.
- **Output**: A boolean value indicating whether the operation was successful (true if the key was found and the length was retrieved, false otherwise).
- **Functions called**:
    - [`gguf_find_key`](../ggml/src/gguf.cpp.driver.md#gguf_find_key)
    - [`format`](llama-impl.cpp.driver.md#format)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::get\_arr<!-- {{#callable:llama_model_loader::get_arr}} -->
The `get_arr` function retrieves an array from a model's metadata using a specified key and stores it in a provided vector, ensuring the array type matches the expected type.
- **Inputs**:
    - `key`: A string representing the key used to locate the array in the model's metadata.
    - `result`: A reference to a vector of type T where the retrieved array will be stored.
    - `required`: A boolean indicating whether the array is required; if true, an exception is thrown if the array is not found.
- **Control Flow**:
    - Finds the key ID using [`gguf_find_key`](../ggml/src/gguf.cpp.driver.md#gguf_find_key) with the provided key.
    - Checks if the key ID is valid and if the key type is `GGUF_TYPE_ARRAY`; if not and `required` is true, throws a runtime error.
    - Retrieves array information using `GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv`.
    - Validates the array type against the expected types (`int32_t`, `uint32_t`, or `float`) using `GGML_ASSERT`.
    - Resizes the result vector to match the array length and assigns the array data to the result vector.
    - Returns true if the array is successfully retrieved and assigned.
- **Output**: Returns a boolean indicating whether the array was successfully retrieved and assigned to the result vector.
- **Functions called**:
    - [`gguf_find_key`](../ggml/src/gguf.cpp.driver.md#gguf_find_key)
    - [`gguf_get_kv_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_kv_type)
    - [`format`](llama-impl.cpp.driver.md#format)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::get\_key<!-- {{#callable:llama_model_loader::get_key}} -->
The `get_key` function retrieves a value associated with a given key from a model's metadata, applying any overrides if present, and throws an error if the key is required but not found.
- **Inputs**:
    - `key`: A string representing the key to look up in the model's metadata.
    - `result`: A reference to a variable of type T where the retrieved value will be stored.
    - `required`: A boolean indicating whether the key is required; if true and the key is not found, an exception is thrown.
- **Control Flow**:
    - Searches for the key in the `kv_overrides` map to check for any overrides.
    - If an override is found, it is used; otherwise, the function attempts to set the result using the `GGUFMeta::GKV<T>::set` method, which retrieves the value from the metadata context.
    - If the key is required and not found, a `std::runtime_error` is thrown with a message indicating the key was not found.
    - Returns a boolean indicating whether the key was found and the result was successfully set.
- **Output**: A boolean value indicating whether the key was found and the result was successfully set.
- **Functions called**:
    - [`format`](llama-impl.cpp.driver.md#format)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::get\_key\_or\_arr<!-- {{#callable:llama_model_loader::get_key_or_arr}} -->
The `get_key_or_arr` function retrieves a key-value pair from a model's metadata, either as an array or a single value repeated, and stores it in a provided array.
- **Inputs**:
    - `key`: A string representing the key to search for in the model's metadata.
    - `result`: A reference to a std::array where the retrieved value(s) will be stored.
    - `n`: A uint32_t representing the number of elements expected to be retrieved.
    - `required`: A boolean indicating whether the key is required to be present in the metadata.
- **Control Flow**:
    - Find the key ID using [`gguf_find_key`](../ggml/src/gguf.cpp.driver.md#gguf_find_key) with the provided key string.
    - If the key ID is not found and the key is required, throw a runtime error; otherwise, return false.
    - Check if the expected number of elements `n` exceeds the maximum size `N_MAX` of the result array; if so, throw a runtime error.
    - Determine if the key corresponds to an array type using [`gguf_get_kv_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_kv_type).
    - If the key is an array, retrieve its metadata using `GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv`.
    - Check if the length of the array matches `n`; if not, throw a runtime error.
    - If the key is an array, call [`get_arr`](#llama_model_loaderget_arr) to populate the result array and return its success status.
    - If the key is not an array, retrieve a single value using [`get_key`](#llama_model_loaderget_key).
    - If the single value retrieval is successful, fill the result array with the value repeated `n` times.
    - Return true if the operation is successful.
- **Output**: A boolean indicating whether the key was successfully retrieved and stored in the result array.
- **Functions called**:
    - [`gguf_find_key`](../ggml/src/gguf.cpp.driver.md#gguf_find_key)
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`gguf_get_kv_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_kv_type)
    - [`llama_model_loader::get_arr`](#llama_model_loaderget_arr)
    - [`llama_model_loader::get_key`](#llama_model_loaderget_key)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::llama\_model\_loader<!-- {{#callable:llama_model_loader::llama_model_loader}} -->
The `llama_model_loader` constructor initializes a model loader by loading model metadata and tensor data from a specified file, handling optional split files, and applying any provided parameter overrides.
- **Inputs**:
    - `fname`: A string representing the filename of the main model file to load.
    - `splits`: A reference to a vector of strings that can optionally contain filenames of split model files.
    - `use_mmap`: A boolean indicating whether to use memory-mapped file access for loading the model.
    - `check_tensors`: A boolean indicating whether to validate tensor data after loading.
    - `param_overrides_p`: A pointer to an array of `llama_model_kv_override` structures for overriding key-value pairs in the model metadata.
    - `param_tensor_buft_overrides_p`: A pointer to an array of `llama_model_tensor_buft_override` structures for overriding tensor buffer types.
- **Control Flow**:
    - Initialize trace level from environment variable `LLAMA_TRACE` if set.
    - Insert key-value overrides from `param_overrides_p` into `kv_overrides` map if provided.
    - Set `tensor_buft_overrides` to `param_tensor_buft_overrides_p`.
    - Initialize GGUF context and load model metadata from the file specified by `fname`.
    - Retrieve and set the model architecture name and key-value type from metadata.
    - Load the main model file and store its context and file handle.
    - Iterate over tensors in the main file, checking for duplicates and storing their metadata in `weights_map`.
    - Retrieve the number of expected split files from metadata and handle additional split files if necessary.
    - For each split file, validate its index, load its context, and store its tensor metadata.
    - Perform a sanity check to ensure the number of loaded tensors matches the expected count.
    - Determine the file type based on tensor quantization types and log metadata information.
    - Check platform support for memory-mapped files and adjust `use_mmap` accordingly.
    - Set instance variables `use_mmap` and `check_tensors` based on input parameters.
- **Output**: The function does not return a value but initializes the `llama_model_loader` object with loaded model data and metadata.
- **Functions called**:
    - [`gguf_init_from_file`](../ggml/src/gguf.cpp.driver.md#gguf_init_from_file)
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`llama_model_loader::get_key`](#llama_model_loaderget_key)
    - [`llm_kv`](llama-arch.h.driver.md#llm_kv)
    - [`LLM_KV`](llama-arch.h.driver.md#LLM_KV)
    - [`llm_arch_from_string`](llama-arch.cpp.driver.md#llm_arch_from_string)
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`ggml_get_name`](../ggml/src/ggml.c.driver.md#ggml_get_name)
    - [`ggml_nelements`](../ggml/src/ggml.c.driver.md#ggml_nelements)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`llama_model_loader::llama_tensor_weight::llama_tensor_weight`](llama-model-loader.h.driver.md#llama_tensor_weightllama_tensor_weight)
    - [`llama_get_list_splits`](#llama_get_list_splits)
    - [`gguf_find_key`](../ggml/src/gguf.cpp.driver.md#gguf_find_key)
    - [`gguf_get_val_u16`](../ggml/src/gguf.cpp.driver.md#gguf_get_val_u16)
    - [`gguf_get_n_kv`](../ggml/src/gguf.cpp.driver.md#gguf_get_n_kv)
    - [`gguf_get_version`](../ggml/src/gguf.cpp.driver.md#gguf_get_version)
    - [`llama_file_version_name`](#llama_file_version_name)
    - [`ggml_type_name`](../ggml/src/ggml.c.driver.md#ggml_type_name)
    - [`gguf_get_key`](../ggml/src/gguf.cpp.driver.md#gguf_get_key)
    - [`gguf_get_kv_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_kv_type)
    - [`gguf_type_name`](../ggml/src/gguf.cpp.driver.md#gguf_type_name)
    - [`gguf_get_arr_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_type)
    - [`gguf_get_arr_n`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_n)
    - [`replace_all`](llama-impl.cpp.driver.md#replace_all)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::get\_arch\_name<!-- {{#callable:llama_model_loader::get_arch_name}} -->
The `get_arch_name` function returns the architecture name of the llama model loader.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of the `arch_name` member variable of the `llama_model_loader` class.
- **Output**: A `std::string` representing the architecture name of the model loader.
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::get\_arch<!-- {{#callable:llama_model_loader::get_arch}} -->
The `get_arch` function returns the architecture type of the llama model as stored in the `llm_kv` member of the `llama_model_loader` class.
- **Inputs**: None
- **Control Flow**:
    - The function directly accesses the `llm_kv` member of the `llama_model_loader` class.
    - It returns the `arch` field from the `llm_kv` object.
- **Output**: The function returns an `enum llm_arch` value representing the architecture type of the model.
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::get\_weight<!-- {{#callable:llama_model_loader::get_weight}} -->
The `get_weight` function retrieves a pointer to a `llama_tensor_weight` object from the `weights_map` based on the provided tensor name.
- **Inputs**:
    - `name`: A C-style string representing the name of the tensor whose weight is to be retrieved.
- **Control Flow**:
    - Searches the `weights_map` for the given `name` using the `find` method.
    - Checks if the search result is not equal to `weights_map.end()`, indicating that the weight was found.
    - If found, returns a pointer to the `llama_tensor_weight` object associated with the name.
    - If not found, returns `nullptr`.
- **Output**: A pointer to the `llama_tensor_weight` object if found, otherwise `nullptr`.
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::require\_weight<!-- {{#callable:llama_model_loader::require_weight}} -->
The `require_weight` function retrieves a tensor weight by name from the `weights_map` and throws an exception if the weight is not found.
- **Inputs**:
    - `name`: A C-style string representing the name of the tensor weight to be retrieved.
- **Control Flow**:
    - Call the [`get_weight`](#llama_model_loaderget_weight) function with the provided `name` to retrieve a pointer to the `llama_tensor_weight` object.
    - Check if the retrieved pointer is null, indicating the weight was not found.
    - If the pointer is null, throw a `std::runtime_error` with a formatted error message indicating the tensor was not found.
    - If the pointer is not null, return the dereferenced `llama_tensor_weight` object.
- **Output**: Returns a reference to the `llama_tensor_weight` object associated with the given name.
- **Functions called**:
    - [`llama_model_loader::get_weight`](#llama_model_loaderget_weight)
    - [`format`](llama-impl.cpp.driver.md#format)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::get\_tensor\_meta<!-- {{#callable:llama_model_loader::get_tensor_meta}} -->
The `get_tensor_meta` function retrieves the metadata of a tensor by its name from a model loader.
- **Inputs**:
    - `name`: A constant character pointer representing the name of the tensor whose metadata is to be retrieved.
- **Control Flow**:
    - Call the [`get_weight`](#llama_model_loaderget_weight) method with the provided tensor name to retrieve the corresponding weight object.
    - Check if the retrieved weight object is null; if it is, return a null pointer.
    - If the weight object is not null, return the `tensor` member of the weight object.
- **Output**: A pointer to a `ggml_tensor` structure representing the tensor metadata, or a null pointer if the tensor is not found.
- **Functions called**:
    - [`llama_model_loader::get_weight`](#llama_model_loaderget_weight)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::require\_tensor\_meta<!-- {{#callable:llama_model_loader::require_tensor_meta}} -->
The `require_tensor_meta` function retrieves the metadata of a tensor by its name and throws an error if the tensor is not found.
- **Inputs**:
    - `name`: A string representing the name of the tensor whose metadata is required.
- **Control Flow**:
    - Call [`get_tensor_meta`](#llama_model_loaderget_tensor_meta) with the provided tensor name to retrieve the tensor metadata.
    - Check if the retrieved tensor is null.
    - If the tensor is null, throw a `std::runtime_error` indicating the tensor was not found.
    - Return the retrieved tensor metadata.
- **Output**: A pointer to the `ggml_tensor` structure representing the metadata of the requested tensor.
- **Functions called**:
    - [`llama_model_loader::get_tensor_meta`](#llama_model_loaderget_tensor_meta)
    - [`format`](llama-impl.cpp.driver.md#format)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::check\_tensor\_dims<!-- {{#callable:llama_model_loader::check_tensor_dims}} -->
The `check_tensor_dims` function verifies if a tensor with a given name has the expected dimensions and returns the tensor metadata if it exists and matches the expected dimensions.
- **Inputs**:
    - `name`: A string representing the name of the tensor to check.
    - `ne`: A vector of int64_t representing the expected dimensions of the tensor.
    - `required`: A boolean indicating whether the tensor is required to exist.
- **Control Flow**:
    - Retrieve the tensor metadata using the [`get_tensor_meta`](#llama_model_loaderget_tensor_meta) function with the provided name.
    - If the tensor metadata is NULL and the tensor is not required, return NULL.
    - If the tensor metadata is NULL and the tensor is required, throw a runtime error indicating the tensor was not found.
    - Initialize a boolean `is_ok` to true to track if the dimensions match.
    - Iterate over the maximum number of dimensions (`GGML_MAX_DIMS`).
    - For each dimension, check if the dimension matches the expected dimension if within the size of `ne`, or is 1 if beyond the size of `ne`.
    - If any dimension does not match, set `is_ok` to false and break the loop.
    - If `is_ok` is false, throw a runtime error indicating the tensor has the wrong shape.
    - Return the tensor metadata.
- **Output**: Returns a pointer to the `ggml_tensor` structure if the tensor exists and matches the expected dimensions, or NULL if the tensor is not required and does not exist.
- **Functions called**:
    - [`llama_model_loader::get_tensor_meta`](#llama_model_loaderget_tensor_meta)
    - [`format`](llama-impl.cpp.driver.md#format)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::create\_tensor<!-- {{#callable:llama_model_loader::create_tensor}} -->
The `create_tensor` function creates a new tensor in a given context, potentially duplicating an existing tensor if specified by flags.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure where the tensor will be created.
    - `name`: A string representing the name of the tensor to be created.
    - `ne`: An initializer list of int64_t values specifying the dimensions of the tensor.
    - `flags`: An integer representing flags that modify the behavior of the function, such as duplication or requirement of the tensor.
- **Control Flow**:
    - The function first calls [`check_tensor_dims`](#llama_model_loadercheck_tensor_dims) to verify the dimensions of the tensor with the given name and dimensions, returning NULL if the tensor is not found and is not required.
    - If the tensor is found, it checks if the `TENSOR_DUPLICATED` flag is set.
    - A new tensor is created by duplicating the existing tensor using [`ggml_dup_tensor`](../ggml/src/ggml.c.driver.md#ggml_dup_tensor).
    - The name of the new tensor is set to the name of the existing tensor using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - If the tensor is duplicated, the size of the data is increased by the number of bytes of the current tensor; otherwise, the count of created tensors is incremented.
    - The newly created tensor is returned.
- **Output**: A pointer to the newly created `ggml_tensor` structure, or NULL if the tensor could not be created.
- **Functions called**:
    - [`llama_model_loader::check_tensor_dims`](#llama_model_loadercheck_tensor_dims)
    - [`ggml_dup_tensor`](../ggml/src/ggml.c.driver.md#ggml_dup_tensor)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
    - [`ggml_get_name`](../ggml/src/ggml.c.driver.md#ggml_get_name)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::create\_tensor\_as\_view<!-- {{#callable:llama_model_loader::create_tensor_as_view}} -->
The `create_tensor_as_view` function creates a new tensor as a view of an existing base tensor with specified dimensions and offset, ensuring type compatibility and updating the tensor count.
- **Inputs**:
    - `ctx`: A pointer to the ggml_context structure, which provides the context for tensor operations.
    - `base`: A pointer to the base ggml_tensor structure from which the new tensor view will be created.
    - `name`: A string representing the name of the new tensor view.
    - `ne`: An initializer list of int64_t specifying the dimensions of the new tensor view.
    - `offset`: A size_t value indicating the offset in the base tensor from which the view starts.
    - `required`: A boolean indicating whether the tensor is required, affecting error handling if the tensor is not found.
- **Control Flow**:
    - Call [`check_tensor_dims`](#llama_model_loadercheck_tensor_dims) to verify the dimensions of the tensor with the given name and dimensions, returning NULL if not found and not required.
    - If the current tensor's type does not match the base tensor's type, throw a runtime error.
    - Initialize an array `dims` to store the dimensions of the new tensor view, filling with 1s for unspecified dimensions.
    - Create a new tensor view using [`ggml_view_4d`](../ggml/src/ggml.c.driver.md#ggml_view_4d) with the specified dimensions and offset, based on the base tensor.
    - Set the name of the new tensor view using [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name).
    - Increment the `n_created` counter to track the number of created tensors.
    - Return the newly created tensor view.
- **Output**: Returns a pointer to the newly created ggml_tensor structure representing the tensor view, or NULL if the tensor is not found and not required.
- **Functions called**:
    - [`llama_model_loader::check_tensor_dims`](#llama_model_loadercheck_tensor_dims)
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`ggml_type_name`](../ggml/src/ggml.c.driver.md#ggml_type_name)
    - [`ggml_view_4d`](../ggml/src/ggml.c.driver.md#ggml_view_4d)
    - [`ggml_set_name`](../ggml/src/ggml.c.driver.md#ggml_set_name)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::done\_getting\_tensors<!-- {{#callable:llama_model_loader::done_getting_tensors}} -->
The `done_getting_tensors` function checks if the number of created tensors matches the expected number and throws an error if they do not match.
- **Inputs**: None
- **Control Flow**:
    - The function checks if `n_created` is not equal to `n_tensors`.
    - If the condition is true, it throws a `std::runtime_error` with a formatted error message indicating the mismatch.
- **Output**: The function does not return any value; it either completes successfully or throws an exception if the condition is not met.
- **Functions called**:
    - [`format`](llama-impl.cpp.driver.md#format)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::init\_mappings<!-- {{#callable:llama_model_loader::init_mappings}} -->
The `init_mappings` function initializes memory mappings for model files and calculates the total size of all tensors for progress reporting.
- **Inputs**:
    - `prefetch`: A boolean indicating whether to prefetch data during memory mapping.
    - `mlock_mmaps`: A pointer to a `llama_mlocks` object, which is used to lock memory mappings if provided.
- **Control Flow**:
    - Check if memory mapping (`use_mmap`) is enabled.
    - Reserve space in `mappings` and `mmaps_used` vectors based on the number of files.
    - Iterate over each file in `files`.
    - For each file, determine if the system is NUMA (Non-Uniform Memory Access) by querying the backend device.
    - Create a new `llama_mmap` object for the file, with prefetching and NUMA settings.
    - Add the size of the mapping to `mmaps_used`.
    - If `mlock_mmaps` is provided, create a `llama_mlock` object, initialize it with the mapping's address, and add it to `mlock_mmaps`.
    - Add the mapping to the `mappings` vector.
    - Iterate over `weights_map` to compute the total size of all tensors and update `size_data`.
- **Output**: The function does not return a value; it modifies the `mappings`, `mmaps_used`, and optionally `mlock_mmaps` data structures, and updates `size_data`.
- **Functions called**:
    - [`ggml_backend_reg_get_proc_address`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::get\_mapping\_range<!-- {{#callable:llama_model_loader::get_mapping_range}} -->
The `get_mapping_range` function determines the range of memory addresses and offsets for a specific tensor mapping in a model, based on the given index and context.
- **Inputs**:
    - `first`: A pointer to a size_t variable where the function will store the smallest offset of the tensor data.
    - `last`: A pointer to a size_t variable where the function will store the largest offset plus the size of the tensor data.
    - `addr`: A pointer to a void pointer where the function will store the base address of the mapping.
    - `idx`: An integer representing the index of the mapping to be used.
    - `ctx`: A pointer to a `ggml_context` object, which provides the context for accessing tensors.
- **Control Flow**:
    - Assert that the `mappings` vector is not empty.
    - Retrieve the mapping at the specified index `idx`.
    - Initialize `first` to the size of the mapping and `last` to 0.
    - Set `addr` to the base address of the mapping.
    - Iterate over all tensors in the given `ctx` context.
    - For each tensor, retrieve its corresponding weight using the tensor's name.
    - If the weight is null or its index does not match `idx`, continue to the next tensor.
    - Update `first` with the minimum of its current value and the weight's offset.
    - Update `last` with the maximum of its current value and the weight's offset plus the tensor's size in bytes.
- **Output**: The function outputs the smallest and largest offsets of the tensor data in the specified mapping, as well as the base address of the mapping.
- **Functions called**:
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`llama_model_loader::get_weight`](#llama_model_loaderget_weight)
    - [`ggml_get_name`](../ggml/src/ggml.c.driver.md#ggml_get_name)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::load\_data\_for<!-- {{#callable:llama_model_loader::load_data_for}} -->
The `load_data_for` function loads tensor data from a file or memory-mapped region into a given tensor structure, validating the data if required.
- **Inputs**:
    - `cur`: A pointer to a `ggml_tensor` structure representing the tensor for which data needs to be loaded.
- **Control Flow**:
    - Retrieve the weight information for the current tensor using [`require_weight`](#llama_model_loaderrequire_weight) with the tensor's name.
    - Check if memory-mapped I/O (`use_mmap`) is enabled.
    - If `use_mmap` is true, get the memory mapping for the weight's index and either set the tensor's data pointer to the mapped address or copy data from the mapped address to the tensor's data.
    - If `use_mmap` is false, assert that the tensor's data is not null and that the weight's index is within the bounds of the files list.
    - Seek to the weight's offset in the corresponding file and read the raw data into the tensor's data buffer.
    - If `check_tensors` is enabled, validate the tensor's data using [`ggml_validate_row_data`](../ggml/src/ggml-quants.c.driver.md#ggml_validate_row_data) and throw a runtime error if the data is invalid.
- **Output**: The function does not return a value; it modifies the `cur` tensor's data in place.
- **Functions called**:
    - [`llama_model_loader::require_weight`](#llama_model_loaderrequire_weight)
    - [`ggml_get_name`](../ggml/src/ggml.c.driver.md#ggml_get_name)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`ggml_validate_row_data`](../ggml/src/ggml-quants.c.driver.md#ggml_validate_row_data)
    - [`format`](llama-impl.cpp.driver.md#format)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::load\_all\_data<!-- {{#callable:llama_model_loader::load_all_data}} -->
The `load_all_data` function loads all tensor data into memory, handling both memory-mapped and non-memory-mapped scenarios, and supports asynchronous uploads to GPU memory if applicable.
- **Inputs**:
    - `ctx`: A pointer to a `ggml_context` structure, representing the context in which tensors are managed.
    - `bufs`: A reference to a `llama_buf_map`, which maps buffer indices to their corresponding buffers.
    - `lmlocks`: A pointer to a `llama_mlocks` structure, used for managing memory locks if applicable.
    - `progress_callback`: A function pointer for a callback to report progress, which can also be used to cancel the operation.
    - `progress_callback_user_data`: A pointer to user data that is passed to the progress callback function.
- **Control Flow**:
    - Assert that `size_data` is not zero, ensuring `init_mappings()` has been called first.
    - Initialize buffers and events for asynchronous uploads if not using memory-mapped I/O.
    - Iterate over each tensor in the context, retrieving its corresponding weight information.
    - For each tensor, check if a progress callback is provided and call it to report progress.
    - Determine the size of the tensor and decide whether to use memory-mapped I/O or direct file reading.
    - If using memory-mapped I/O, map the tensor data directly or allocate it in a buffer if necessary.
    - If not using memory-mapped I/O, read the tensor data from the file, either directly or using asynchronous uploads if supported.
    - Update the total size of data processed.
    - Free any temporary resources used for asynchronous uploads.
    - Check the results of any asynchronous validation tasks and throw an error if any tensor data is invalid.
    - If all data has been loaded, perform final cleanup, including unmapping memory-mapped regions if applicable.
    - Return `true` if all data is successfully loaded, or `false` if the operation is canceled via the progress callback.
- **Output**: Returns a boolean indicating whether all data was successfully loaded (`true`) or if the operation was canceled (`false`).
- **Functions called**:
    - [`ggml_backend_buft_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buft_name)
    - [`ggml_backend_dev_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_name)
    - [`ggml_backend_dev_get_props`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_get_props)
    - [`ggml_backend_buffer_get_base`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_get_base)
    - [`ggml_backend_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_name)
    - [`ggml_get_first_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_first_tensor)
    - [`ggml_get_next_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_next_tensor)
    - [`llama_model_loader::get_weight`](#llama_model_loaderget_weight)
    - [`ggml_get_name`](../ggml/src/ggml.c.driver.md#ggml_get_name)
    - [`ggml_nbytes`](../ggml/src/ggml.c.driver.md#ggml_nbytes)
    - [`ggml_validate_row_data`](../ggml/src/ggml-quants.c.driver.md#ggml_validate_row_data)
    - [`ggml_backend_tensor_alloc`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_alloc)
    - [`ggml_backend_tensor_set`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set)
    - [`ggml_backend_buffer_is_host`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_is_host)
    - [`ggml_backend_event_synchronize`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_event_synchronize)
    - [`ggml_backend_tensor_set_async`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_tensor_set_async)
    - [`ggml_backend_event_record`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_event_record)
    - [`format`](llama-impl.cpp.driver.md#format)
    - [`ggml_backend_event_free`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_event_free)
    - [`ggml_backend_buffer_free`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buffer_free)
    - [`ggml_backend_free`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_free)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::ftype\_name<!-- {{#callable:llama_model_loader::ftype_name}} -->
The `ftype_name` function returns the name of the file type associated with the llama model loader.
- **Inputs**: None
- **Control Flow**:
    - The function calls [`llama_model_ftype_name`](#llama_model_ftype_name) with the `ftype` member variable of the `llama_model_loader` class.
    - It returns the result of the [`llama_model_ftype_name`](#llama_model_ftype_name) function call.
- **Output**: A `std::string` representing the name of the file type.
- **Functions called**:
    - [`llama_model_ftype_name`](#llama_model_ftype_name)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)


---
#### llama\_model\_loader::print\_info<!-- {{#callable:llama_model_loader::print_info}} -->
The `print_info` method logs detailed information about the file format, file type, and file size of the model being loaded.
- **Inputs**: None
- **Control Flow**:
    - Log the file format using [`llama_file_version_name`](#llama_file_version_name) to get the version name from `fver`.
    - Log the file type using [`llama_model_ftype_name`](#llama_model_ftype_name) to get the type name from `ftype`.
    - Check if `n_bytes` is less than a gigabyte; if true, log the file size in megabytes and bits per word (BPW).
    - If `n_bytes` is greater than or equal to a gigabyte, log the file size in gigabytes and BPW.
- **Output**: The function does not return any value; it outputs information to the log.
- **Functions called**:
    - [`llama_file_version_name`](#llama_file_version_name)
    - [`llama_model_ftype_name`](#llama_model_ftype_name)
- **See also**: [`llama_model_loader`](llama-model-loader.h.driver.md#llama_model_loader)  (Data Structure)



# Functions

---
### llama\_file\_version\_name<!-- {{#callable:llama_file_version_name}} -->
The function `llama_file_version_name` returns a string representation of a given file version identifier for the GGUF file format.
- **Inputs**:
    - `version`: An enumeration value of type `llama_fver` representing the file version.
- **Control Flow**:
    - The function uses a switch statement to match the input `version` with predefined cases.
    - If the `version` matches `GGUF_FILE_VERSION_V1`, it returns the string "GGUF V1 (support until nov 2023)".
    - If the `version` matches `GGUF_FILE_VERSION_V2`, it returns the string "GGUF V2".
    - If the `version` matches `GGUF_FILE_VERSION_V3`, it returns the string "GGUF V3 (latest)".
    - If the `version` does not match any of the predefined cases, it returns the string "unknown".
- **Output**: A constant character pointer to a string describing the file version, or "unknown" if the version is not recognized.


---
### llama\_model\_ftype\_name<!-- {{#callable:llama_model_ftype_name}} -->
The function `llama_model_ftype_name` returns a string representation of a given llama model file type, optionally indicating if the type was guessed.
- **Inputs**:
    - `ftype`: An enumeration value of type `llama_ftype` representing the file type of a llama model.
- **Control Flow**:
    - Check if the `ftype` has the `LLAMA_FTYPE_GUESSED` flag set.
    - If guessed, recursively call `llama_model_ftype_name` with the guessed flag removed and append ' (guessed)' to the result.
    - Use a switch statement to match the `ftype` to its corresponding string representation.
    - Return the string representation for the matched `ftype`.
    - If no match is found, return 'unknown, may not work'.
- **Output**: A string that describes the file type of the llama model, with an optional ' (guessed)' suffix if the type was guessed.


---
### llama\_get\_list\_splits<!-- {{#callable:llama_get_list_splits}} -->
The function `llama_get_list_splits` generates a list of file paths for split files based on a given path, index, and number of splits.
- **Inputs**:
    - `path`: A string representing the base path of the split files.
    - `idx`: An integer representing the index of the current split file.
    - `n_split`: An integer representing the total number of split files.
- **Control Flow**:
    - Initialize an empty vector `paths` to store the resulting file paths.
    - Initialize a string `split_prefix` and a character buffer `buf` with a size determined by `llama_path_max()`.
    - Call `llama_split_prefix` to generate a prefix for the split files and store it in `split_prefix`; throw an error if the prefix is invalid.
    - Iterate over the range from 0 to `n_split`, calling `llama_split_path` to generate each split file path and append it to `paths`.
    - Return the vector `paths` containing all the split file paths.
- **Output**: A vector of strings, each representing a path to a split file.
- **Functions called**:
    - [`llama_path_max`](llama-mmap.cpp.driver.md#llama_path_max)
    - [`format`](llama-impl.cpp.driver.md#format)


---
### getter<!-- {{#callable:GGUFMeta::getter}} -->
The `getter` function retrieves array information from a given context and key, returning an `ArrayInfo` structure with the array type, length, and data pointer.
- **Inputs**:
    - `ctx`: A pointer to a `gguf_context` structure, representing the context from which the array information is to be retrieved.
    - `k`: An integer representing the key or index of the array within the context.
- **Control Flow**:
    - Retrieve the array type using [`gguf_get_arr_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_type) with the provided context and key.
    - Retrieve the array length using [`gguf_get_arr_n`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_n) with the provided context and key.
    - Determine the data pointer: if the array type is `GGUF_TYPE_STRING`, set the data pointer to `nullptr`; otherwise, retrieve the data pointer using [`gguf_get_arr_data`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_data).
    - Return an `ArrayInfo` structure initialized with the array type, length, and data pointer.
- **Output**: An `ArrayInfo` structure containing the array type, length, and data pointer.
- **Functions called**:
    - [`gguf_get_arr_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_type)
    - [`gguf_get_arr_n`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_n)
    - [`gguf_get_arr_data`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_data)


