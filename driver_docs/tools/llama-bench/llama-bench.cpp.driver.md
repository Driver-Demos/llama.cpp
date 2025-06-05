# Purpose
The provided C++ source code is a comprehensive benchmarking tool designed to evaluate the performance of machine learning models, specifically those related to the "llama" framework. This code is structured to handle various configurations and parameters, allowing users to test different aspects of model performance, such as prompt processing and token generation. The code includes functionality for parsing command-line arguments, setting up test parameters, and executing tests with different configurations. It supports multiple output formats, including CSV, JSON, JSONL, Markdown, and SQL, to record and display the results of the benchmarks.

Key components of the code include the definition of command-line parameters, the setup of test instances, and the execution of tests using the llama framework. The code utilizes a variety of C++ standard library features, such as templates, vectors, and maps, to manage data and configurations efficiently. It also includes utility functions for time measurement, string manipulation, and data transformation. The code is designed to be extensible and configurable, allowing users to specify different models, batch sizes, thread counts, and other parameters to tailor the benchmarking process to their specific needs. The inclusion of detailed logging and error handling ensures that users can diagnose issues and understand the performance characteristics of their models.
# Imports and Dependencies

---
- `algorithm`
- `array`
- `cassert`
- `chrono`
- `cinttypes`
- `clocale`
- `cmath`
- `cstdio`
- `cstdlib`
- `cstring`
- `ctime`
- `iterator`
- `map`
- `numeric`
- `regex`
- `sstream`
- `string`
- `thread`
- `vector`
- `common.h`
- `ggml.h`
- `llama.h`
- `windows.h`


# Global Variables

---
### cmd\_params\_defaults
- **Type**: ``cmd_params``
- **Description**: The `cmd_params_defaults` variable is a static constant instance of the `cmd_params` struct, which holds default configuration values for various command-line parameters used in the application. It includes settings for model file paths, prompt and generation sizes, batch sizes, data types, threading, GPU layers, and other performance-related options.
- **Use**: This variable is used to provide default values for command-line parameters, ensuring consistent initial settings for the application's execution.


---
### build\_commit
- **Type**: ``std::string``
- **Description**: The `build_commit` is a global constant string variable defined within the `test` namespace. It is initialized with the value of `LLAMA_COMMIT`, which is likely a macro or constant representing the commit hash or identifier of the build.
- **Use**: This variable is used to store and provide the commit identifier of the build for reference or logging purposes.


---
### build\_number
- **Type**: ``const int``
- **Description**: The `build_number` is a constant integer variable defined within the `test` namespace. It is initialized with the value of `LLAMA_BUILD_NUMBER`, which is likely a macro or constant defined elsewhere in the code.
- **Use**: This variable is used to store the build number of the software, which can be useful for version tracking and debugging purposes.


# Data Structures

---
### output\_formats<!-- {{#data_structure:output_formats}} -->
- **Type**: `enum`
- **Members**:
    - `NONE`: Represents no output format.
    - `CSV`: Represents the CSV output format.
    - `JSON`: Represents the JSON output format.
    - `JSONL`: Represents the JSONL output format.
    - `MARKDOWN`: Represents the Markdown output format.
    - `SQL`: Represents the SQL output format.
- **Description**: The `output_formats` enum defines a set of constants representing different output formats that can be used in the application. These formats include NONE, CSV, JSON, JSONL, MARKDOWN, and SQL, allowing the application to specify how data should be formatted when outputting results.


---
### cmd\_params<!-- {{#data_structure:cmd_params}} -->
- **Type**: `struct`
- **Members**:
    - `model`: A vector of strings representing model file paths.
    - `n_prompt`: A vector of integers representing the number of prompts.
    - `n_gen`: A vector of integers representing the number of generations.
    - `n_pg`: A vector of pairs of integers representing prompt-generation pairs.
    - `n_depth`: A vector of integers representing the depth of operations.
    - `n_batch`: A vector of integers representing the batch size.
    - `n_ubatch`: A vector of integers representing the unbatch size.
    - `type_k`: A vector of ggml_type representing the type for key cache.
    - `type_v`: A vector of ggml_type representing the type for value cache.
    - `defrag_thold`: A vector of floats representing the defragmentation threshold.
    - `n_threads`: A vector of integers representing the number of threads.
    - `cpu_mask`: A vector of strings representing CPU masks.
    - `cpu_strict`: A vector of booleans indicating strict CPU usage.
    - `poll`: A vector of integers representing polling intervals.
    - `n_gpu_layers`: A vector of integers representing the number of GPU layers.
    - `rpc_servers`: A vector of strings representing RPC server addresses.
    - `split_mode`: A vector of llama_split_mode representing the split mode.
    - `main_gpu`: A vector of integers representing the main GPU index.
    - `no_kv_offload`: A vector of booleans indicating if key-value offloading is disabled.
    - `flash_attn`: A vector of booleans indicating if flash attention is enabled.
    - `tensor_split`: A vector of vectors of floats representing tensor split ratios.
    - `tensor_buft_overrides`: A vector of vectors of llama_model_tensor_buft_override representing tensor buffer overrides.
    - `use_mmap`: A vector of booleans indicating if memory-mapped files are used.
    - `embeddings`: A vector of booleans indicating if embeddings are used.
    - `no_op_offload`: A vector of booleans indicating if operation offloading is disabled.
    - `numa`: A ggml_numa_strategy representing the NUMA strategy.
    - `reps`: An integer representing the number of repetitions for tests.
    - `prio`: A ggml_sched_priority representing the scheduling priority.
    - `delay`: An integer representing the delay between tests.
    - `verbose`: A boolean indicating if verbose output is enabled.
    - `progress`: A boolean indicating if progress indicators are shown.
    - `output_format`: An output_formats enum representing the output format for stdout.
    - `output_format_stderr`: An output_formats enum representing the output format for stderr.
- **Description**: The `cmd_params` struct is a comprehensive configuration structure used to store various parameters for command-line operations, particularly in the context of machine learning model testing and benchmarking. It includes vectors for multiple configurations such as model paths, prompt and generation counts, batch sizes, and types for caching. Additionally, it handles CPU and GPU settings, including thread counts, CPU masks, and GPU layer configurations. The struct also supports advanced features like NUMA strategies, scheduling priorities, and output formatting options, making it highly versatile for performance testing and optimization tasks.


---
### cmd\_params\_instance<!-- {{#data_structure:cmd_params_instance}} -->
- **Type**: `struct`
- **Members**:
    - `model`: A string representing the model name or path.
    - `n_prompt`: An integer specifying the number of prompts.
    - `n_gen`: An integer specifying the number of generations.
    - `n_depth`: An integer specifying the depth of the model.
    - `n_batch`: An integer specifying the batch size.
    - `n_ubatch`: An integer specifying the unbatch size.
    - `type_k`: A ggml_type specifying the type for key tensors.
    - `type_v`: A ggml_type specifying the type for value tensors.
    - `defrag_thold`: A float representing the defragmentation threshold.
    - `n_threads`: An integer specifying the number of threads.
    - `cpu_mask`: A string representing the CPU mask.
    - `cpu_strict`: A boolean indicating if CPU strict mode is enabled.
    - `poll`: An integer specifying the polling interval.
    - `n_gpu_layers`: An integer specifying the number of GPU layers.
    - `rpc_servers_str`: A string containing RPC server addresses.
    - `split_mode`: A llama_split_mode specifying the split mode.
    - `main_gpu`: An integer specifying the main GPU index.
    - `no_kv_offload`: A boolean indicating if key-value offloading is disabled.
    - `flash_attn`: A boolean indicating if flash attention is enabled.
    - `tensor_split`: A vector of floats representing tensor split ratios.
    - `tensor_buft_overrides`: A vector of llama_model_tensor_buft_override for tensor buffer overrides.
    - `use_mmap`: A boolean indicating if memory-mapped files are used.
    - `embeddings`: A boolean indicating if embeddings are used.
    - `no_op_offload`: A boolean indicating if operation offloading is disabled.
- **Description**: The `cmd_params_instance` struct is a comprehensive configuration data structure used to define various parameters for a command execution context, particularly in the context of machine learning model operations. It includes fields for model configuration, execution parameters like batch sizes and thread counts, and hardware-specific settings such as GPU layers and CPU masks. This struct is designed to facilitate the setup and execution of model operations by encapsulating all necessary parameters in a single entity, allowing for easy manipulation and transfer of configuration settings.
- **Member Functions**:
    - [`cmd_params_instance::to_llama_mparams`](#cmd_params_instanceto_llama_mparams)
    - [`cmd_params_instance::equal_mparams`](#cmd_params_instanceequal_mparams)
    - [`cmd_params_instance::to_llama_cparams`](#cmd_params_instanceto_llama_cparams)

**Methods**

---
#### cmd\_params\_instance::to\_llama\_mparams<!-- {{#callable:cmd_params_instance::to_llama_mparams}} -->
The `to_llama_mparams` function converts the current instance of `cmd_params_instance` into a `llama_model_params` structure, configuring it with various parameters including GPU layers, RPC servers, split mode, and tensor buffer overrides.
- **Inputs**: None
- **Control Flow**:
    - Initialize `llama_model_params` with default parameters using `llama_model_default_params()`.
    - Set `mparams.n_gpu_layers` to the instance's `n_gpu_layers`.
    - If `rpc_servers_str` is not empty, split it into individual server strings and attempt to register RPC devices.
    - Check if the RPC backend is available and retrieve the function to add RPC devices; exit with an error if not found.
    - For each server in `rpc_servers`, attempt to add it as an RPC device; exit with an error if any addition fails.
    - Set `mparams.devices` to the list of added devices.
    - Set `mparams.split_mode`, `mparams.main_gpu`, `mparams.tensor_split`, and `mparams.use_mmap` from the instance's corresponding fields.
    - If `tensor_buft_overrides` is empty, set `mparams.tensor_buft_overrides` to `nullptr`; otherwise, ensure it is properly terminated and set it to the instance's data.
    - Return the configured `llama_model_params` structure.
- **Output**: Returns a `llama_model_params` structure configured with the instance's parameters.
- **Functions called**:
    - [`llama_model_default_params`](../../src/llama-model.cpp.driver.md#llama_model_default_params)
    - [`ggml_backend_reg_get_proc_address`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address)
- **See also**: [`cmd_params_instance`](#cmd_params_instance)  (Data Structure)


---
#### cmd\_params\_instance::equal\_mparams<!-- {{#callable:cmd_params_instance::equal_mparams}} -->
The `equal_mparams` function checks if two `cmd_params_instance` objects have equal model parameters by comparing specific attributes.
- **Inputs**:
    - `other`: A reference to another `cmd_params_instance` object to compare against the current instance.
- **Control Flow**:
    - The function compares the `model` attribute of the current instance with the `model` attribute of the `other` instance.
    - It compares the `n_gpu_layers` attribute of the current instance with the `n_gpu_layers` attribute of the `other` instance.
    - It compares the `rpc_servers_str` attribute of the current instance with the `rpc_servers_str` attribute of the `other` instance.
    - It compares the `split_mode` attribute of the current instance with the `split_mode` attribute of the `other` instance.
    - It compares the `main_gpu` attribute of the current instance with the `main_gpu` attribute of the `other` instance.
    - It compares the `use_mmap` attribute of the current instance with the `use_mmap` attribute of the `other` instance.
    - It compares the `tensor_split` vector of the current instance with the `tensor_split` vector of the `other` instance.
    - It calls the [`vec_tensor_buft_override_equal`](#vec_tensor_buft_override_equal) function to compare the `tensor_buft_overrides` vector of the current instance with the `tensor_buft_overrides` vector of the `other` instance.
    - If all comparisons return true, the function returns true; otherwise, it returns false.
- **Output**: A boolean value indicating whether the two `cmd_params_instance` objects have equal model parameters.
- **Functions called**:
    - [`vec_tensor_buft_override_equal`](#vec_tensor_buft_override_equal)
- **See also**: [`cmd_params_instance`](#cmd_params_instance)  (Data Structure)


---
#### cmd\_params\_instance::to\_llama\_cparams<!-- {{#callable:cmd_params_instance::to_llama_cparams}} -->
The `to_llama_cparams` function initializes and returns a `llama_context_params` object with specific parameters set based on the current instance of `cmd_params_instance`.
- **Inputs**: None
- **Control Flow**:
    - Initialize a `llama_context_params` object `cparams` using `llama_context_default_params()`.
    - Set `cparams.n_ctx` to the sum of `n_prompt`, `n_gen`, and `n_depth`.
    - Assign `n_batch` to `cparams.n_batch`.
    - Assign `n_ubatch` to `cparams.n_ubatch`.
    - Set `cparams.type_k` to `type_k`.
    - Set `cparams.type_v` to `type_v`.
    - Assign `defrag_thold` to `cparams.defrag_thold`.
    - Set `cparams.offload_kqv` to the negation of `no_kv_offload`.
    - Assign `flash_attn` to `cparams.flash_attn`.
    - Assign `embeddings` to `cparams.embeddings`.
    - Set `cparams.op_offload` to the negation of `no_op_offload`.
    - Set `cparams.swa_full` to `false`.
    - Return the configured `cparams` object.
- **Output**: A `llama_context_params` object with parameters set based on the instance's attributes.
- **Functions called**:
    - [`llama_context_default_params`](../../src/llama-context.cpp.driver.md#llama_context_default_params)
- **See also**: [`cmd_params_instance`](#cmd_params_instance)  (Data Structure)



---
### test<!-- {{#data_structure:test}} -->
- **Type**: `struct`
- **Members**:
    - `build_commit`: A static constant string representing the build commit identifier.
    - `build_number`: A static constant integer representing the build number.
    - `cpu_info`: A constant string containing information about the CPU.
    - `gpu_info`: A constant string containing information about the GPU.
    - `model_filename`: A string representing the filename of the model.
    - `model_type`: A string representing the type of the model.
    - `model_size`: A 64-bit unsigned integer representing the size of the model.
    - `model_n_params`: A 64-bit unsigned integer representing the number of parameters in the model.
    - `n_batch`: An integer representing the batch size.
    - `n_ubatch`: An integer representing the unbatch size.
    - `n_threads`: An integer representing the number of threads.
    - `cpu_mask`: A string representing the CPU mask.
    - `cpu_strict`: A boolean indicating if strict CPU usage is enforced.
    - `poll`: An integer representing the polling interval.
    - `type_k`: A ggml_type representing the type of key.
    - `type_v`: A ggml_type representing the type of value.
    - `defrag_thold`: A float representing the defragmentation threshold.
    - `n_gpu_layers`: An integer representing the number of GPU layers.
    - `split_mode`: A llama_split_mode representing the split mode.
    - `main_gpu`: An integer representing the main GPU index.
    - `no_kv_offload`: A boolean indicating if key-value offloading is disabled.
    - `flash_attn`: A boolean indicating if flash attention is enabled.
    - `tensor_split`: A vector of floats representing the tensor split configuration.
    - `tensor_buft_overrides`: A vector of llama_model_tensor_buft_override representing tensor buffer overrides.
    - `use_mmap`: A boolean indicating if memory-mapped files are used.
    - `embeddings`: A boolean indicating if embeddings are used.
    - `no_op_offload`: A boolean indicating if operation offloading is disabled.
    - `n_prompt`: An integer representing the number of prompts.
    - `n_gen`: An integer representing the number of generations.
    - `n_depth`: An integer representing the depth.
    - `test_time`: A string representing the test time in RFC 3339 format.
    - `samples_ns`: A vector of 64-bit unsigned integers representing sample times in nanoseconds.
- **Description**: The `test` struct is a comprehensive data structure designed to encapsulate various parameters and metadata related to a model testing environment. It includes static constants for build information, and a wide array of fields for storing CPU and GPU information, model details, threading and batching configurations, and various operational flags. The struct is initialized with parameters from a `cmd_params_instance` and a model context, and it provides methods for calculating average and standard deviation of sample times, as well as retrieving backend information and field values. This struct is integral to managing and executing model tests, capturing performance metrics, and facilitating the configuration of test environments.
- **Member Functions**:
    - [`test::test`](#testtest)
    - [`test::avg_ns`](#testavg_ns)
    - [`test::stdev_ns`](#teststdev_ns)
    - [`test::get_ts`](#testget_ts)
    - [`test::avg_ts`](#testavg_ts)
    - [`test::stdev_ts`](#teststdev_ts)
    - [`test::get_backend`](#testget_backend)
    - [`test::get_fields`](#testget_fields)
    - [`test::get_field_type`](#testget_field_type)
    - [`test::get_values`](#testget_values)
    - [`test::get_map`](#testget_map)

**Methods**

---
#### test::test<!-- {{#callable:test::test}} -->
The `test` constructor initializes a `test` object with various parameters from a `cmd_params_instance` and a `llama_model`, capturing system information and setting up model-related attributes.
- **Inputs**:
    - `inst`: A `cmd_params_instance` object containing various configuration parameters for the test.
    - `lmodel`: A pointer to a `llama_model` object from which model-specific information is extracted.
    - `ctx`: A pointer to a `llama_context` object, which is not used in the constructor but is part of the signature.
- **Control Flow**:
    - Initialize `cpu_info` and `gpu_info` by calling `get_cpu_info()` and `get_gpu_info()` respectively.
    - Set `model_filename` to the model name from `inst`.
    - Use `llama_model_desc` to get a description of the model and store it in `model_type`.
    - Retrieve model size and number of parameters using `llama_model_size` and `llama_model_n_params`, and store them in `model_size` and `model_n_params`.
    - Assign various parameters from `inst` to corresponding member variables of the `test` object.
    - Get the current time, format it in RFC 3339 format, and store it in `test_time`.
    - The `ctx` parameter is cast to void to avoid unused parameter warnings.
- **Output**: The function does not return a value; it initializes a `test` object with the provided parameters and system information.
- **Functions called**:
    - [`get_cpu_info`](#get_cpu_info)
    - [`get_gpu_info`](#get_gpu_info)
- **See also**: [`test`](#test)  (Data Structure)


---
#### test::avg\_ns<!-- {{#callable:test::avg_ns}} -->
The `avg_ns` function calculates the average of the elements in the `samples_ns` vector.
- **Inputs**: None
- **Control Flow**:
    - The function calls the global `avg` function, passing `samples_ns` as an argument.
    - The `avg` function computes the average of the elements in the `samples_ns` vector and returns it.
- **Output**: The function returns a `uint64_t` representing the average of the `samples_ns` vector elements.
- **See also**: [`test`](#test)  (Data Structure)


---
#### test::stdev\_ns<!-- {{#callable:test::stdev_ns}} -->
The `stdev_ns` function calculates the standard deviation of a vector of sample times in nanoseconds.
- **Inputs**:
    - `samples_ns`: A vector of uint64_t values representing sample times in nanoseconds.
- **Control Flow**:
    - The function calls the global `stdev` function, passing `samples_ns` as an argument.
    - The `stdev` function computes the standard deviation of the values in the vector.
- **Output**: Returns a uint64_t value representing the standard deviation of the sample times in nanoseconds.
- **See also**: [`test`](#test)  (Data Structure)


---
#### test::get\_ts<!-- {{#callable:test::get_ts}} -->
The `get_ts` function calculates and returns a vector of throughput values in tokens per second based on the recorded sample times in nanoseconds.
- **Inputs**:
    - `n_prompt`: An integer representing the number of prompt tokens.
    - `n_gen`: An integer representing the number of generated tokens.
    - `samples_ns`: A vector of uint64_t values representing sample times in nanoseconds.
- **Control Flow**:
    - Calculate the total number of tokens by summing `n_prompt` and `n_gen`.
    - Initialize an empty vector `ts` to store throughput values.
    - Use `std::transform` to iterate over `samples_ns`, applying a lambda function to each element.
    - The lambda function calculates the throughput as `1e9 * n_tokens / t` for each sample time `t`, converting nanoseconds to seconds and dividing by the total number of tokens.
    - Append the calculated throughput values to the `ts` vector.
    - Return the `ts` vector containing the throughput values.
- **Output**: A vector of double values representing the throughput in tokens per second for each sample time.
- **See also**: [`test`](#test)  (Data Structure)


---
#### test::avg\_ts<!-- {{#callable:test::avg_ts}} -->
The `avg_ts` function calculates the average throughput in tokens per second based on the time samples collected during a test.
- **Inputs**: None
- **Control Flow**:
    - The function calls `get_ts()` to retrieve a vector of throughput values calculated from `samples_ns`.
    - It then calls the global `avg` function with this vector to compute the average throughput.
- **Output**: Returns a `double` representing the average throughput in tokens per second.
- **Functions called**:
    - [`test::get_ts`](#testget_ts)
- **See also**: [`test`](#test)  (Data Structure)


---
#### test::stdev\_ts<!-- {{#callable:test::stdev_ts}} -->
The `stdev_ts` function calculates the standard deviation of the transformed sample times in the `test` structure.
- **Inputs**: None
- **Control Flow**:
    - The function calls `get_ts()` to retrieve a vector of transformed sample times.
    - It then calls the global `stdev` function with this vector to compute the standard deviation.
- **Output**: Returns a `double` representing the standard deviation of the transformed sample times.
- **Functions called**:
    - [`test::get_ts`](#testget_ts)
- **See also**: [`test`](#test)  (Data Structure)


---
#### test::get\_backend<!-- {{#callable:test::get_backend}} -->
The `get_backend` function retrieves a list of available backends, excluding 'CPU', and returns them as a comma-separated string, defaulting to 'CPU' if no other backends are found.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty vector `backends` to store backend names.
    - Iterate over the number of registered backends using `ggml_backend_reg_count()`.
    - For each backend, retrieve its registration using `ggml_backend_reg_get(i)` and its name using `ggml_backend_reg_name(reg)`.
    - If the backend name is not 'CPU', add it to the `backends` vector.
    - After the loop, check if `backends` is empty; if so, return 'CPU'.
    - If `backends` is not empty, join the names in `backends` with a comma and return the resulting string.
- **Output**: A string representing the available backends, excluding 'CPU', joined by commas, or 'CPU' if no other backends are available.
- **Functions called**:
    - [`ggml_backend_reg_count`](../../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_reg_count)
    - [`ggml_backend_reg_name`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_name)
    - [`join`](#join)
- **See also**: [`test`](#test)  (Data Structure)


---
#### test::get\_fields<!-- {{#callable:test::get_fields}} -->
The `get_fields` function returns a static reference to a vector of strings representing various field names related to a test structure.
- **Inputs**: None
- **Control Flow**:
    - The function defines a static vector of strings named `fields` containing field names such as 'build_commit', 'build_number', 'cpu_info', etc.
    - The function returns a reference to this static vector.
- **Output**: A constant reference to a vector of strings containing field names.
- **See also**: [`test`](#test)  (Data Structure)


---
#### test::get\_field\_type<!-- {{#callable:test::get_field_type}} -->
The `get_field_type` function determines the data type of a given field name from a predefined set of fields and returns the corresponding `field_type` enum value.
- **Inputs**:
    - `field`: A constant reference to a `std::string` representing the name of the field whose type is to be determined.
- **Control Flow**:
    - The function checks if the input field name matches any of the predefined field names associated with the `INT` type and returns `INT` if a match is found.
    - If no match is found for `INT`, it checks if the field name matches any of the predefined field names associated with the `BOOL` type and returns `BOOL` if a match is found.
    - If no match is found for `BOOL`, it checks if the field name matches any of the predefined field names associated with the `FLOAT` type and returns `FLOAT` if a match is found.
    - If the field name does not match any of the predefined field names for `INT`, `BOOL`, or `FLOAT`, the function returns `STRING` as the default type.
- **Output**: The function returns a `field_type` enum value, which can be `INT`, `BOOL`, `FLOAT`, or `STRING`, indicating the data type of the specified field.
- **See also**: [`test`](#test)  (Data Structure)


---
#### test::get\_values<!-- {{#callable:test::get_values}} -->
The `get_values` function constructs and returns a vector of strings representing various attributes and configurations of a test object, including formatted tensor split and buffer override information.
- **Inputs**: None
- **Control Flow**:
    - Initialize two empty strings `tensor_split_str` and `tensor_buft_overrides_str`, and an integer `max_nonzero` to 0.
    - Iterate over `tensor_split` to find the maximum index `max_nonzero` where the value is greater than 0.
    - Format each non-zero value in `tensor_split` up to `max_nonzero` into a string with two decimal places, appending it to `tensor_split_str` with '/' separators.
    - Check if `tensor_buft_overrides` has only one element and assert it is a null pattern, appending 'none' to `tensor_buft_overrides_str` if true.
    - Otherwise, iterate over `tensor_buft_overrides` (excluding the last null pattern) to construct a string representation of each override, appending 'none' or formatted pattern and buffer type to `tensor_buft_overrides_str` with ';' separators.
    - Construct a vector `values` containing various attributes of the test object, including the formatted `tensor_split_str` and `tensor_buft_overrides_str`.
    - Return the `values` vector.
- **Output**: A `std::vector<std::string>` containing string representations of various attributes and configurations of the test object.
- **Functions called**:
    - [`ggml_backend_buft_name`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buft_name)
    - [`test::get_backend`](#testget_backend)
    - [`ggml_type_name`](../../ggml/src/ggml.c.driver.md#ggml_type_name)
    - [`split_mode_str`](#split_mode_str)
    - [`test::avg_ns`](#testavg_ns)
    - [`test::stdev_ns`](#teststdev_ns)
    - [`test::avg_ts`](#testavg_ts)
    - [`test::stdev_ts`](#teststdev_ts)
- **See also**: [`test`](#test)  (Data Structure)


---
#### test::get\_map<!-- {{#callable:test::get_map}} -->
The `get_map` function constructs and returns a map that associates field names with their corresponding values from the `test` structure.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty map of type `std::map<std::string, std::string>`.
    - Retrieve the list of field names by calling `get_fields()`.
    - Retrieve the list of field values by calling `get_values()`.
    - Use `std::transform` to iterate over the fields and values simultaneously, inserting each field-value pair into the map using `std::make_pair`.
    - Return the constructed map.
- **Output**: A `std::map<std::string, std::string>` where each key is a field name and each value is the corresponding field value from the `test` structure.
- **Functions called**:
    - [`test::get_fields`](#testget_fields)
    - [`test::get_values`](#testget_values)
- **See also**: [`test`](#test)  (Data Structure)



---
### field\_type<!-- {{#data_structure:test::field_type}} -->
- **Type**: `enum`
- **Members**:
    - `STRING`: Represents a string field type.
    - `BOOL`: Represents a boolean field type.
    - `INT`: Represents an integer field type.
    - `FLOAT`: Represents a floating-point field type.
- **Description**: The `field_type` enum defines a set of constants representing different types of fields that can be used in a data structure or application. It includes four types: STRING, BOOL, INT, and FLOAT, which correspond to string, boolean, integer, and floating-point data types, respectively. This enum is useful for categorizing or identifying the type of data a field is expected to hold.


---
### printer<!-- {{#data_structure:printer}} -->
- **Type**: `struct`
- **Members**:
    - `fout`: A pointer to a FILE object used for output.
- **Description**: The `printer` struct is an abstract base class designed for outputting test results in various formats. It contains a virtual destructor and three virtual methods: `print_header`, `print_test`, and `print_footer`, which are intended to be overridden by derived classes to implement specific output formats. The `fout` member is a pointer to a FILE object, which is used to direct the output to a specific file or stream.
- **Member Functions**:
    - [`printer::~printer`](#printerprinter)
    - [`printer::print_header`](#printerprint_header)
    - [`printer::print_footer`](#printerprint_footer)

**Methods**

---
#### printer::\~printer<!-- {{#callable:printer::~printer}} -->
The `~printer` function is a virtual destructor for the `printer` class, ensuring proper cleanup of derived class objects.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as a virtual destructor, allowing derived class destructors to be called correctly when an object is deleted through a base class pointer.
    - The function body is empty, indicating no specific cleanup is required for the `printer` class itself.
- **Output**: The function does not return any value.
- **See also**: [`printer`](#printer)  (Data Structure)


---
#### printer::print\_header<!-- {{#callable:printer::print_header}} -->
The `print_header` function is a virtual method in the `printer` struct that takes a `cmd_params` object as input and does nothing with it.
- **Inputs**:
    - `params`: A `cmd_params` object containing command parameters, which is not used in the function.
- **Control Flow**:
    - The function is defined as a virtual method in the `printer` struct.
    - It takes a single argument of type `cmd_params` by reference.
    - The function body contains a single statement that casts the `params` argument to void, effectively ignoring it.
- **Output**: The function does not produce any output or perform any operations.
- **See also**: [`printer`](#printer)  (Data Structure)


---
#### printer::print\_footer<!-- {{#callable:printer::print_footer}} -->
The `print_footer` function is a virtual method in the `printer` class that is intended to be overridden by derived classes to print a footer, but it does nothing in its base implementation.
- **Inputs**: None
- **Control Flow**:
    - The function is defined as a virtual method, allowing derived classes to override it.
    - The function body is empty, indicating no operations are performed in the base class implementation.
- **Output**: There is no output from this function as it is an empty method.
- **See also**: [`printer`](#printer)  (Data Structure)



---
### csv\_printer<!-- {{#data_structure:csv_printer}} -->
- **Type**: `struct`
- **Description**: The `csv_printer` struct is a specialized printer that inherits from the `printer` base class. It is designed to output data in CSV format. The struct includes a static method `escape_csv` to handle CSV-specific escaping of fields, ensuring that any quotes within the data are properly escaped. The `print_header` method outputs the CSV header using field names obtained from the `test` class, while the `print_test` method outputs the test data values, applying CSV escaping to each value before printing. This struct is part of a larger system for benchmarking and outputting test results in various formats.
- **Member Functions**:
    - [`csv_printer::escape_csv`](#csv_printerescape_csv)
    - [`csv_printer::print_header`](#csv_printerprint_header)
    - [`csv_printer::print_test`](#csv_printerprint_test)
- **Inherits From**:
    - [`printer`](#printer)

**Methods**

---
#### csv\_printer::escape\_csv<!-- {{#callable:csv_printer::escape_csv}} -->
The `escape_csv` function escapes a given string for safe inclusion in a CSV file by surrounding it with double quotes and doubling any internal double quotes.
- **Inputs**:
    - `field`: A constant reference to a `std::string` representing the field to be escaped for CSV.
- **Control Flow**:
    - Initialize an empty string `escaped` with a starting double quote.
    - Iterate over each character `c` in the input string `field`.
    - If the character `c` is a double quote, append another double quote to `escaped` to escape it.
    - Append the character `c` to `escaped`.
    - After the loop, append a closing double quote to `escaped`.
    - Return the `escaped` string.
- **Output**: A `std::string` that is the escaped version of the input `field`, suitable for CSV formatting.
- **See also**: [`csv_printer`](#csv_printer)  (Data Structure)


---
#### csv\_printer::print\_header<!-- {{#callable:csv_printer::print_header}} -->
The `print_header` function outputs a CSV header line to a file stream using field names obtained from a test structure.
- **Inputs**:
    - `params`: A constant reference to a `cmd_params` object, which is not used in the function body.
- **Control Flow**:
    - Retrieve a vector of field names by calling `test::get_fields()`.
    - Join the field names into a single string separated by commas using the [`join`](#join) function.
    - Print the joined string followed by a newline to the file stream `fout` using `fprintf`.
    - Explicitly ignore the `params` argument by casting it to void.
- **Output**: The function does not return any value; it outputs a CSV header line to the file stream `fout`.
- **Functions called**:
    - [`join`](#join)
- **See also**: [`csv_printer`](#csv_printer)  (Data Structure)


---
#### csv\_printer::print\_test<!-- {{#callable:csv_printer::print_test}} -->
The `print_test` function formats and prints the values of a `test` object as a CSV line to a file stream.
- **Inputs**:
    - `t`: A `test` object whose values are to be printed.
- **Control Flow**:
    - Retrieve the values from the `test` object `t` using `t.get_values()`.
    - Transform each value in the vector to escape CSV special characters using `escape_csv`.
    - Join the transformed values into a single CSV line using `join(values, ",")`.
    - Print the resulting CSV line to the file stream `fout` using `fprintf`.
- **Output**: The function does not return a value; it outputs a formatted CSV line to the file stream `fout`.
- **Functions called**:
    - [`join`](#join)
- **See also**: [`csv_printer`](#csv_printer)  (Data Structure)



---
### json\_printer<!-- {{#data_structure:json_printer}} -->
- **Type**: `struct`
- **Members**:
    - `first`: A boolean flag indicating if the first JSON object is being printed.
- **Description**: The `json_printer` struct is a specialized printer for outputting test results in JSON format. It inherits from the `printer` base class and overrides methods to print the header, individual test results, and the footer in a JSON array format. The `first` member is used to manage the formatting of the JSON output, ensuring that commas are correctly placed between JSON objects. This struct is designed to handle the conversion of test data into a JSON-friendly format, including escaping strings and formatting values according to their types.
- **Member Functions**:
    - [`json_printer::print_header`](#json_printerprint_header)
    - [`json_printer::print_fields`](#json_printerprint_fields)
    - [`json_printer::print_test`](#json_printerprint_test)
    - [`json_printer::print_footer`](#json_printerprint_footer)
- **Inherits From**:
    - [`printer`](#printer)

**Methods**

---
#### json\_printer::print\_header<!-- {{#callable:json_printer::print_header}} -->
The `print_header` function writes the opening bracket of a JSON array to the output file stream.
- **Inputs**:
    - `params`: A constant reference to a `cmd_params` object, which contains command parameters, but is not used in the function.
- **Control Flow**:
    - The function begins by writing the string "[\n" to the file stream `fout`, indicating the start of a JSON array.
    - The `params` argument is explicitly cast to void to indicate it is unused in the function.
- **Output**: The function does not return any value; it writes directly to the file stream `fout`.
- **See also**: [`json_printer`](#json_printer)  (Data Structure)


---
#### json\_printer::print\_fields<!-- {{#callable:json_printer::print_fields}} -->
The `print_fields` function outputs a formatted JSON-like representation of field-value pairs to a file stream.
- **Inputs**:
    - `fields`: A constant reference to a vector of strings representing the field names.
    - `values`: A constant reference to a vector of strings representing the corresponding values for each field.
- **Control Flow**:
    - The function asserts that the size of the `fields` vector is equal to the size of the `values` vector to ensure each field has a corresponding value.
    - It iterates over the `fields` vector using a for loop, accessing each field and its corresponding value by index.
    - For each field-value pair, it formats the pair as a JSON-like string and writes it to the file stream `fout` using `fprintf`.
- **Output**: The function does not return a value; it writes formatted output to a file stream.
- **Functions called**:
    - [`format_json_value`](#format_json_value)
- **See also**: [`json_printer`](#json_printer)  (Data Structure)


---
#### json\_printer::print\_test<!-- {{#callable:json_printer::print_test}} -->
The `print_test` function formats and outputs test data in JSON format to a file stream.
- **Inputs**:
    - `t`: A constant reference to a `test` object containing the data to be printed.
- **Control Flow**:
    - Check if this is the first test being printed; if not, print a comma and newline to separate JSON objects.
    - Print the opening brace for the JSON object.
    - Call [`print_fields`](#json_printerprint_fields) to print the fields and their values from the `test` object.
    - Print the `samples_ns` field with its values formatted as a JSON array.
    - Print the `samples_ts` field with its values formatted as a JSON array.
    - Print the closing brace for the JSON object.
    - Flush the output stream to ensure all data is written.
- **Output**: The function outputs formatted JSON data to the file stream `fout`.
- **Functions called**:
    - [`json_printer::print_fields`](#json_printerprint_fields)
    - [`join`](#join)
- **See also**: [`json_printer`](#json_printer)  (Data Structure)


---
#### json\_printer::print\_footer<!-- {{#callable:json_printer::print_footer}} -->
The `print_footer` function outputs the closing bracket of a JSON array to a file stream.
- **Inputs**: None
- **Control Flow**:
    - The function uses `fprintf` to write a newline followed by a closing square bracket and another newline to the file stream `fout`.
- **Output**: The function does not return any value; it writes directly to the file stream `fout`.
- **See also**: [`json_printer`](#json_printer)  (Data Structure)



---
### jsonl\_printer<!-- {{#data_structure:jsonl_printer}} -->
- **Type**: `struct`
- **Description**: The `jsonl_printer` struct is a specialized printer that inherits from the `printer` base class. It is designed to output test results in JSON Lines (JSONL) format, which is a convenient format for streaming JSON objects. The struct overrides the `print_test` method to format and print the fields and values of a test object as a JSON object, followed by a newline character. This allows each test result to be a separate line of JSON, making it easy to process large datasets line-by-line.
- **Member Functions**:
    - [`jsonl_printer::print_fields`](#jsonl_printerprint_fields)
    - [`jsonl_printer::print_test`](#jsonl_printerprint_test)
- **Inherits From**:
    - [`printer`](#printer)

**Methods**

---
#### jsonl\_printer::print\_fields<!-- {{#callable:jsonl_printer::print_fields}} -->
The `print_fields` function outputs formatted JSON-like key-value pairs to a file stream, ensuring the fields and values vectors are of equal size.
- **Inputs**:
    - `fields`: A constant reference to a vector of strings representing the field names.
    - `values`: A constant reference to a vector of strings representing the corresponding values for each field.
- **Control Flow**:
    - The function begins by asserting that the size of the `fields` vector is equal to the size of the `values` vector, ensuring they are paired correctly.
    - It then iterates over each element in the `fields` vector using a for loop.
    - For each iteration, it retrieves the current field and value, formats them into a JSON-like string using the [`format_json_value`](#format_json_value) function, and writes the formatted string to the `fout` file stream using `fprintf`.
- **Output**: The function does not return a value; it outputs formatted strings directly to the `fout` file stream.
- **Functions called**:
    - [`format_json_value`](#format_json_value)
- **See also**: [`jsonl_printer`](#jsonl_printer)  (Data Structure)


---
#### jsonl\_printer::print\_test<!-- {{#callable:jsonl_printer::print_test}} -->
The `print_test` function outputs a JSON-like representation of a `test` object to a file stream.
- **Inputs**:
    - `t`: A `test` object containing various fields and sample data to be printed.
- **Control Flow**:
    - The function begins by printing an opening curly brace to the file stream `fout`.
    - It calls [`print_fields`](#json_printerprint_fields) to print the fields and their values from the `test` object.
    - It prints the `samples_ns` field as a JSON array by joining the `samples_ns` vector with commas.
    - It prints the `samples_ts` field as a JSON array by joining the result of `t.get_ts()` with commas.
    - The function ends by printing a closing curly brace and a newline character, then flushes the output stream.
- **Output**: The function does not return a value; it writes the formatted output to the file stream `fout`.
- **Functions called**:
    - [`json_printer::print_fields`](#json_printerprint_fields)
    - [`join`](#join)
- **See also**: [`jsonl_printer`](#jsonl_printer)  (Data Structure)



---
### markdown\_printer<!-- {{#data_structure:markdown_printer}} -->
- **Type**: `struct`
- **Members**:
    - `fields`: A vector of strings representing the fields to be printed in the markdown format.
- **Description**: The `markdown_printer` struct is a specialized printer that formats and outputs data in a markdown table format. It inherits from the `printer` base class and overrides methods to print headers, test results, and footers specifically for markdown output. The struct maintains a list of fields to be printed, and provides static methods to determine field widths and display names for formatting purposes. It is designed to handle various field types and adjust the table layout accordingly, ensuring that the output is well-structured and readable in markdown format.
- **Member Functions**:
    - [`markdown_printer::get_field_width`](#markdown_printerget_field_width)
    - [`markdown_printer::get_field_display_name`](#markdown_printerget_field_display_name)
    - [`markdown_printer::print_header`](#markdown_printerprint_header)
    - [`markdown_printer::print_test`](#markdown_printerprint_test)
    - [`markdown_printer::print_footer`](#markdown_printerprint_footer)
- **Inherits From**:
    - [`printer`](#printer)

**Methods**

---
#### markdown\_printer::get\_field\_width<!-- {{#callable:markdown_printer::get_field_width}} -->
The `get_field_width` function determines the width of a field for display purposes based on the field's name and type.
- **Inputs**:
    - `field`: A constant reference to a `std::string` representing the name of the field whose width is to be determined.
- **Control Flow**:
    - Check if the field name matches specific predefined strings and return corresponding predefined widths.
    - If the field name does not match any predefined strings, calculate the width as the maximum of the field's length and 10.
    - Check the field type using `test::get_field_type(field)`; if it is a string, return the negative of the calculated width.
    - Return the calculated width.
- **Output**: An integer representing the width of the field, which may be negative if the field is a string type.
- **See also**: [`markdown_printer`](#markdown_printer)  (Data Structure)


---
#### markdown\_printer::get\_field\_display\_name<!-- {{#callable:markdown_printer::get_field_display_name}} -->
The `get_field_display_name` function returns a shortened display name for a given field name, or the field name itself if no abbreviation is defined.
- **Inputs**:
    - `field`: A string representing the name of the field for which a display name is to be retrieved.
- **Control Flow**:
    - The function checks if the input field matches any predefined field names like 'n_gpu_layers', 'split_mode', etc.
    - If a match is found, it returns the corresponding abbreviated display name such as 'ngl', 'sm', etc.
    - If no match is found, it returns the input field name as is.
- **Output**: A string that is either the abbreviated display name of the field or the field name itself if no abbreviation is defined.
- **See also**: [`markdown_printer`](#markdown_printer)  (Data Structure)


---
#### markdown\_printer::print\_header<!-- {{#callable:markdown_printer::print_header}} -->
The `print_header` function dynamically selects and prints a formatted header row for a markdown table based on the provided command parameters and backend configuration.
- **Inputs**:
    - `params`: A `cmd_params` structure containing various command parameters that influence which fields are included in the header.
- **Control Flow**:
    - Initialize a list of fields with default entries: 'model', 'size', 'params', and 'backend'.
    - Determine if the backend is CPU-based by checking if 'CPU' or 'BLAS' is present in the backend string.
    - Conditionally add 'n_gpu_layers' to fields if the backend is not CPU-based.
    - Iterate over various parameters in `params`, comparing each to its default value or checking its size, and add corresponding fields to the list if conditions are met.
    - Add 'test' and 't/s' to the fields list unconditionally.
    - Print the header row by iterating over the selected fields, formatting each field name according to its width and display name.
    - Print a separator row beneath the header, using dashes and alignment indicators based on field widths.
- **Output**: The function outputs a formatted markdown table header to the file stream `fout`, which includes selected fields based on the input parameters and backend configuration.
- **Functions called**:
    - [`vec_vec_tensor_buft_override_equal`](#vec_vec_tensor_buft_override_equal)
    - [`markdown_printer::get_field_width`](#markdown_printerget_field_width)
    - [`markdown_printer::get_field_display_name`](#markdown_printerget_field_display_name)
- **See also**: [`markdown_printer`](#markdown_printer)  (Data Structure)


---
#### markdown\_printer::print\_test<!-- {{#callable:markdown_printer::print_test}} -->
The `print_test` function formats and prints the details of a test object in a tabular format based on specified fields.
- **Inputs**:
    - `t`: A `test` object containing various attributes related to a model test, such as model type, size, parameters, backend, and performance metrics.
- **Control Flow**:
    - Retrieve a map of field-value pairs from the test object using `t.get_map()`.
    - Iterate over each field in the `fields` vector.
    - For each field, determine the appropriate value to display based on the field name, using conditional logic to format values like model size, parameters, and test performance metrics.
    - Calculate the width for each field using [`get_field_width`](#markdown_printerget_field_width) and adjust for UTF-8 characters if necessary.
    - Print each field's value in a formatted manner using `fprintf`, ensuring alignment within the table.
    - End the line with a newline character after all fields are printed.
- **Output**: The function outputs a formatted line of text to the file stream `fout`, representing the test object's data in a table row.
- **Functions called**:
    - [`markdown_printer::get_field_width`](#markdown_printerget_field_width)
- **See also**: [`markdown_printer`](#markdown_printer)  (Data Structure)


---
#### markdown\_printer::print\_footer<!-- {{#callable:markdown_printer::print_footer}} -->
The `print_footer` function outputs the build commit and build number to a file stream.
- **Inputs**: None
- **Control Flow**:
    - The function uses `fprintf` to write a formatted string to the `fout` file stream.
    - It retrieves the build commit and build number from the `test` namespace and formats them into the string.
- **Output**: The function outputs a formatted string containing the build commit and build number to the `fout` file stream.
- **See also**: [`markdown_printer`](#markdown_printer)  (Data Structure)



---
### sql\_printer<!-- {{#data_structure:sql_printer}} -->
- **Type**: ``struct``
- **Description**: The `sql_printer` struct is a specialized printer that inherits from the `printer` base class. It is designed to output SQL commands for creating tables and inserting data into a database. The struct includes methods for determining the SQL field type based on a given field's type, printing the SQL table creation header, and inserting test data into the table. It does not have any member variables, focusing instead on its functionality to format and output SQL statements.
- **Member Functions**:
    - [`sql_printer::get_sql_field_type`](#sql_printerget_sql_field_type)
    - [`sql_printer::print_header`](#sql_printerprint_header)
    - [`sql_printer::print_test`](#sql_printerprint_test)
- **Inherits From**:
    - [`printer`](#printer)

**Methods**

---
#### sql\_printer::get\_sql\_field\_type<!-- {{#callable:sql_printer::get_sql_field_type}} -->
The `get_sql_field_type` function maps a field type from a custom enumeration to a corresponding SQL data type as a string.
- **Inputs**:
    - `field`: A constant reference to a `std::string` representing the name of the field whose SQL type is to be determined.
- **Control Flow**:
    - The function calls `test::get_field_type(field)` to determine the field type of the input `field`.
    - A switch statement is used to map the field type to a corresponding SQL data type string.
    - If the field type is `test::STRING`, the function returns "TEXT".
    - If the field type is `test::BOOL` or `test::INT`, the function returns "INTEGER".
    - If the field type is `test::FLOAT`, the function returns "REAL".
    - If the field type does not match any of the specified cases, the function triggers an assertion failure and exits the program.
- **Output**: A `std::string` representing the SQL data type corresponding to the input field's type.
- **See also**: [`sql_printer`](#sql_printer)  (Data Structure)


---
#### sql\_printer::print\_header<!-- {{#callable:sql_printer::print_header}} -->
The `print_header` function generates and outputs a SQL `CREATE TABLE` statement for a table named 'test' with fields and their corresponding SQL data types.
- **Inputs**:
    - `params`: A constant reference to a `cmd_params` object, which is not used in the function body.
- **Control Flow**:
    - Retrieve a list of field names by calling `test::get_fields()` and store them in a vector `fields`.
    - Output the beginning of a SQL `CREATE TABLE` statement to the file stream `fout`.
    - Iterate over each field in the `fields` vector.
    - For each field, determine its SQL data type using `get_sql_field_type()` and output the field name and type to `fout`, appending a comma if it is not the last field.
    - Output the closing parenthesis and semicolon to complete the SQL statement.
    - Output a newline character to `fout`.
    - Explicitly ignore the `params` argument by casting it to void.
- **Output**: The function outputs a SQL `CREATE TABLE` statement to the file stream `fout`.
- **Functions called**:
    - [`sql_printer::get_sql_field_type`](#sql_printerget_sql_field_type)
- **See also**: [`sql_printer`](#sql_printer)  (Data Structure)


---
#### sql\_printer::print\_test<!-- {{#callable:sql_printer::print_test}} -->
The `print_test` function generates and outputs an SQL INSERT statement for a given `test` object, inserting its field values into a database table.
- **Inputs**:
    - `t`: A `test` object containing various fields and values to be inserted into the SQL database.
- **Control Flow**:
    - The function begins by printing the SQL INSERT statement header with the table name and field names using `fprintf`.
    - It retrieves the values from the `test` object using `t.get_values()` and stores them in a vector.
    - A loop iterates over the values, printing each value enclosed in single quotes and separated by commas, except for the last value which is not followed by a comma.
    - Finally, the function prints the closing parenthesis and semicolon to complete the SQL statement.
- **Output**: The function does not return a value; it outputs the SQL INSERT statement directly to the file stream `fout`.
- **Functions called**:
    - [`join`](#join)
- **See also**: [`sql_printer`](#sql_printer)  (Data Structure)



# Functions

---
### get\_time\_ns<!-- {{#callable:get_time_ns}} -->
The `get_time_ns` function retrieves the current time in nanoseconds since the epoch using a high-resolution clock.
- **Inputs**: None
- **Control Flow**:
    - The function defines an alias `clock` for `std::chrono::high_resolution_clock`.
    - It calls `clock::now()` to get the current time point.
    - It calculates the duration since the epoch in nanoseconds using `std::chrono::nanoseconds(clock::now().time_since_epoch())`.
    - It returns the count of nanoseconds as a `uint64_t`.
- **Output**: The function returns the current time in nanoseconds as a `uint64_t`.


---
### tensor\_buft\_override\_equal<!-- {{#callable:tensor_buft_override_equal}} -->
The function `tensor_buft_override_equal` checks if two `llama_model_tensor_buft_override` objects are equal by comparing their `pattern` and `buft` attributes.
- **Inputs**:
    - `a`: The first `llama_model_tensor_buft_override` object to compare.
    - `b`: The second `llama_model_tensor_buft_override` object to compare.
- **Control Flow**:
    - Check if the `pattern` attributes of `a` and `b` are not equal.
    - If either `pattern` is `nullptr`, return `false`.
    - If `pattern` attributes are not `nullptr` and not equal (using `strcmp`), return `false`.
    - Check if the `buft` attributes of `a` and `b` are not equal, return `false` if they are not.
    - If all checks pass, return `true`.
- **Output**: Returns `true` if both `llama_model_tensor_buft_override` objects are equal, otherwise `false`.


---
### vec\_tensor\_buft\_override\_equal<!-- {{#callable:vec_tensor_buft_override_equal}} -->
The function `vec_tensor_buft_override_equal` checks if two vectors of `llama_model_tensor_buft_override` objects are equal by comparing their sizes and elements.
- **Inputs**:
    - `a`: A vector of `llama_model_tensor_buft_override` objects to be compared.
    - `b`: Another vector of `llama_model_tensor_buft_override` objects to be compared.
- **Control Flow**:
    - Check if the sizes of vectors `a` and `b` are different; if so, return `false`.
    - Iterate over each element in the vectors `a` and `b`.
    - For each pair of elements, call [`tensor_buft_override_equal`](#tensor_buft_override_equal) to check if they are equal.
    - If any pair of elements is not equal, return `false`.
    - If all elements are equal, return `true`.
- **Output**: A boolean value indicating whether the two vectors are equal.
- **Functions called**:
    - [`tensor_buft_override_equal`](#tensor_buft_override_equal)


---
### vec\_vec\_tensor\_buft\_override\_equal<!-- {{#callable:vec_vec_tensor_buft_override_equal}} -->
The function `vec_vec_tensor_buft_override_equal` checks if two 2D vectors of `llama_model_tensor_buft_override` objects are equal by comparing their sizes and elements.
- **Inputs**:
    - `a`: A 2D vector of `llama_model_tensor_buft_override` objects to be compared.
    - `b`: Another 2D vector of `llama_model_tensor_buft_override` objects to be compared.
- **Control Flow**:
    - Check if the sizes of vectors `a` and `b` are different; if so, return false.
    - Iterate over each vector in `a` and `b` using a loop.
    - For each pair of vectors from `a` and `b`, call [`vec_tensor_buft_override_equal`](#vec_tensor_buft_override_equal) to check if they are equal.
    - If any pair of vectors is not equal, return false.
    - If all pairs are equal, return true.
- **Output**: Returns a boolean value indicating whether the two 2D vectors are equal.
- **Functions called**:
    - [`vec_tensor_buft_override_equal`](#vec_tensor_buft_override_equal)


---
### join<!-- {{#callable:join}} -->
The `join` function concatenates elements of a vector into a single string, separated by a specified delimiter.
- **Inputs**:
    - `values`: A constant reference to a vector of elements of type T, which are to be joined into a string.
    - `delim`: A constant reference to a string that serves as the delimiter between elements in the resulting string.
- **Control Flow**:
    - Initialize an output string stream `str`.
    - Iterate over each element in the `values` vector.
    - For each element, append it to the string stream `str`.
    - If the current element is not the last one, append the delimiter `delim` to the string stream.
    - After the loop, convert the string stream to a string and return it.
- **Output**: A string that contains all elements of the input vector concatenated together, separated by the specified delimiter.


---
### transform\_to\_str<!-- {{#callable:transform_to_str}} -->
The `transform_to_str` function transforms a vector of any type into a vector of strings using a provided transformation function.
- **Inputs**:
    - `values`: A constant reference to a vector of elements of type `T` that need to be transformed into strings.
    - `f`: A transformation function or functor that takes an element of type `T` and returns a `std::string`.
- **Control Flow**:
    - Initialize an empty vector `str_values` to store the resulting strings.
    - Use `std::transform` to apply the transformation function `f` to each element in the input vector `values`, inserting the results into `str_values`.
    - Return the `str_values` vector containing the transformed strings.
- **Output**: A vector of strings, where each string is the result of applying the transformation function `f` to the corresponding element in the input vector `values`.


---
### avg<!-- {{#callable:avg}} -->
The `avg` function calculates the average of elements in a given vector.
- **Inputs**:
    - `v`: A constant reference to a vector of elements of type T, where T is a template parameter.
- **Control Flow**:
    - Check if the vector `v` is empty; if so, return 0.
    - Use `std::accumulate` to calculate the sum of all elements in the vector `v`, starting with an initial value of T(0).
    - Divide the calculated sum by the size of the vector `v` to get the average.
    - Return the calculated average.
- **Output**: The function returns the average of the elements in the vector as a value of type T.


---
### stdev<!-- {{#callable:stdev}} -->
The `stdev` function calculates the standard deviation of a vector of numeric values.
- **Inputs**:
    - `v`: A constant reference to a vector of numeric values of type T.
- **Control Flow**:
    - Check if the size of the vector is less than or equal to 1; if so, return 0 as the standard deviation is not defined for such cases.
    - Calculate the mean of the vector using the [`avg`](#avg) function.
    - Compute the sum of squares of the vector elements using `std::inner_product`.
    - Calculate the standard deviation using the formula: sqrt((sum of squares / (n-1)) - (mean^2 * n / (n-1))), where n is the size of the vector.
    - Return the calculated standard deviation.
- **Output**: The function returns the standard deviation of the vector as a value of type T.
- **Functions called**:
    - [`avg`](#avg)


---
### get\_cpu\_info<!-- {{#callable:get_cpu_info}} -->
The `get_cpu_info` function retrieves and returns a comma-separated string of descriptions for all CPU and accelerator devices available in the backend.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty vector `cpu_list` to store CPU descriptions.
    - Iterate over all devices using `ggml_backend_dev_count()` to get the total number of devices.
    - For each device, retrieve the device pointer using `ggml_backend_dev_get(i)`.
    - Determine the device type using `ggml_backend_dev_type(dev)`.
    - If the device type is `GGML_BACKEND_DEVICE_TYPE_CPU` or `GGML_BACKEND_DEVICE_TYPE_ACCEL`, append its description to `cpu_list` using `ggml_backend_dev_description(dev)`.
    - Join all descriptions in `cpu_list` into a single string separated by commas using the [`join`](#join) function.
    - Return the resulting string.
- **Output**: A string containing comma-separated descriptions of all CPU and accelerator devices.
- **Functions called**:
    - [`ggml_backend_dev_count`](../../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_dev_count)
    - [`ggml_backend_dev_type`](../../ggml/include/ggml-backend.h.driver.md#ggml_backend_dev_type)
    - [`ggml_backend_dev_description`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_description)
    - [`join`](#join)


---
### get\_gpu\_info<!-- {{#callable:get_gpu_info}} -->
The `get_gpu_info` function retrieves and returns a comma-separated string of descriptions for all GPU devices available in the backend.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty vector `gpu_list` to store GPU descriptions.
    - Iterate over all devices using `ggml_backend_dev_count()` to get the total number of devices.
    - For each device, retrieve the device pointer using `ggml_backend_dev_get(i)`.
    - Determine the device type using `ggml_backend_dev_type(dev)`.
    - If the device type is `GGML_BACKEND_DEVICE_TYPE_GPU`, append its description to `gpu_list` using `ggml_backend_dev_description(dev)`.
    - After iterating through all devices, join the descriptions in `gpu_list` into a single string separated by commas using the [`join`](#join) function.
    - Return the resulting string.
- **Output**: A `std::string` containing a comma-separated list of descriptions for all GPU devices.
- **Functions called**:
    - [`ggml_backend_dev_count`](../../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_dev_count)
    - [`ggml_backend_dev_type`](../../ggml/include/ggml-backend.h.driver.md#ggml_backend_dev_type)
    - [`ggml_backend_dev_description`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_description)
    - [`join`](#join)


---
### output\_format\_str<!-- {{#callable:output_format_str}} -->
The `output_format_str` function returns a string representation of an `output_formats` enum value.
- **Inputs**:
    - `format`: An `output_formats` enum value representing the desired output format.
- **Control Flow**:
    - The function uses a switch statement to match the `format` argument against predefined enum values.
    - For each case (NONE, CSV, JSON, JSONL, MARKDOWN, SQL), it returns a corresponding string representation.
    - If the `format` does not match any case, the function calls `GGML_ABORT` with an error message indicating an invalid output format.
- **Output**: A constant character pointer to a string representing the output format.


---
### output\_format\_from\_str<!-- {{#callable:output_format_from_str}} -->
The function `output_format_from_str` converts a string representation of an output format to its corresponding enum value and returns a boolean indicating success.
- **Inputs**:
    - `s`: A constant reference to a `std::string` representing the name of the output format.
    - `format`: A reference to an `output_formats` enum variable where the corresponding enum value will be stored.
- **Control Flow**:
    - The function checks if the input string `s` matches any of the predefined format strings: "none", "csv", "json", "jsonl", "md", or "sql".
    - If a match is found, the corresponding enum value (NONE, CSV, JSON, JSONL, MARKDOWN, or SQL) is assigned to `format`.
    - If no match is found, the function returns `false`.
    - If a match is found, the function returns `true`.
- **Output**: A boolean value indicating whether the conversion was successful (`true` if a match was found, `false` otherwise).


---
### split\_mode\_str<!-- {{#callable:split_mode_str}} -->
The `split_mode_str` function returns a string representation of a given `llama_split_mode` enumeration value.
- **Inputs**:
    - `mode`: An enumeration value of type `llama_split_mode` which specifies the split mode to be converted to a string.
- **Control Flow**:
    - The function uses a switch statement to determine the string representation of the `mode` argument.
    - If `mode` is `LLAMA_SPLIT_MODE_NONE`, the function returns the string "none".
    - If `mode` is `LLAMA_SPLIT_MODE_LAYER`, the function returns the string "layer".
    - If `mode` is `LLAMA_SPLIT_MODE_ROW`, the function returns the string "row".
    - If `mode` does not match any of the specified cases, the function calls `GGML_ABORT` with the message "invalid split mode".
- **Output**: A constant character pointer to the string representation of the `llama_split_mode` value.


---
### pair\_str<!-- {{#callable:pair_str}} -->
The `pair_str` function converts a `std::pair<int, int>` to a comma-separated string representation.
- **Inputs**:
    - `p`: A constant reference to a `std::pair<int, int>` object, representing a pair of integers to be converted to a string.
- **Control Flow**:
    - Declare a static character buffer `buf` of size 32.
    - Use `snprintf` to format the integers `p.first` and `p.second` into the buffer `buf` as a comma-separated string.
    - Return the formatted string stored in `buf`.
- **Output**: A `std::string` containing the comma-separated representation of the integer pair.


---
### parse\_int\_range<!-- {{#callable:parse_int_range}} -->
The `parse_int_range` function parses a string representing a range of integers and returns a vector of integers that fall within that range.
- **Inputs**:
    - `s`: A string representing a range of integers, which can be in the format 'first', 'first-last', 'first-last+step', or 'first-last*step'.
- **Control Flow**:
    - Initialize a regex pattern to match the range format and prepare a string iterator for searching.
    - While there are matches in the string, extract the first, last, operator, and step values from the match.
    - If 'last' is not specified, set it equal to 'first'.
    - If 'operator' is not specified, default it to '+'.
    - If 'step' is not specified, default it to 1.
    - Iterate from 'first' to 'last', applying the operator with the step to generate the range of integers.
    - Add each integer to the result vector.
    - If the operator is invalid or the range is invalid (e.g., step results in no progress), throw an invalid_argument exception.
    - After processing all matches, if there is any unmatched part of the string, throw an invalid_argument exception.
- **Output**: A vector of integers representing the parsed range.


---
### print\_usage<!-- {{#callable:print_usage}} -->
The `print_usage` function displays the usage instructions and available command-line options for a program.
- **Inputs**:
    - `argc`: This parameter is not used in the function, indicated by the comment `/* argc */`.
    - `argv`: An array of C-style strings representing the command-line arguments passed to the program, where `argv[0]` is the program name.
- **Control Flow**:
    - The function begins by printing the basic usage format using `argv[0]` to include the program name.
    - It then prints a list of available options, each with a brief description and default values where applicable.
    - The options include flags for help, NUMA mode, repetitions, priority, delay, output formats, verbosity, and progress indicators.
    - It also lists test parameters such as model filename, prompt size, generation size, batch sizes, cache types, thread count, CPU mask, GPU layers, and other advanced settings.
    - The function checks if RPC is supported and conditionally prints the RPC option.
    - Finally, it provides additional information on how to specify multiple values and ranges for parameters.
- **Output**: The function does not return any value; it outputs the usage information directly to the standard output.
- **Functions called**:
    - [`output_format_str`](#output_format_str)
    - [`join`](#join)
    - [`transform_to_str`](#transform_to_str)


---
### ggml\_type\_from\_name<!-- {{#callable:ggml_type_from_name}} -->
The function `ggml_type_from_name` maps a string representation of a type to its corresponding `ggml_type` enumeration value.
- **Inputs**:
    - `s`: A constant reference to a `std::string` representing the name of the type.
- **Control Flow**:
    - The function checks if the input string `s` matches any of the predefined type names such as "f16", "bf16", "q8_0", etc.
    - For each match, it returns the corresponding `ggml_type` enumeration value like `GGML_TYPE_F16`, `GGML_TYPE_BF16`, `GGML_TYPE_Q8_0`, etc.
    - If none of the predefined type names match the input string, the function returns `GGML_TYPE_COUNT`.
- **Output**: The function returns a `ggml_type` enumeration value corresponding to the input string, or `GGML_TYPE_COUNT` if the string does not match any known type.


---
### parse\_cmd\_params<!-- {{#callable:parse_cmd_params}} -->
The `parse_cmd_params` function parses command-line arguments to configure various parameters for a program, setting defaults and handling errors as needed.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize a `cmd_params` structure with default values from `cmd_params_defaults`.
    - Iterate over each command-line argument starting from index 1.
    - For each argument, check if it matches a known option and parse the associated value(s) if applicable.
    - If an argument is invalid or missing a required value, set `invalid_param` to true and break the loop.
    - If `invalid_param` is true after parsing, print an error message and usage instructions, then exit the program.
    - After parsing, set any parameters that are still empty to their default values from `cmd_params_defaults`.
- **Output**: Returns a `cmd_params` structure populated with the parsed command-line parameters.
- **Functions called**:
    - [`print_usage`](#print_usage)
    - [`parse_int_range`](#parse_int_range)
    - [`ggml_type_from_name`](#ggml_type_from_name)
    - [`ggml_backend_dev_count`](../../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_dev_count)
    - [`ggml_backend_buft_name`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buft_name)
    - [`output_format_from_str`](#output_format_from_str)


---
### get\_cmd\_params\_instances<!-- {{#callable:get_cmd_params_instances}} -->
The function `get_cmd_params_instances` generates a list of `cmd_params_instance` objects based on the combinations of parameters provided in a `cmd_params` object.
- **Inputs**:
    - `params`: A `cmd_params` object containing various configuration parameters, each represented as a vector of possible values.
- **Control Flow**:
    - Initialize an empty vector `instances` to store `cmd_params_instance` objects.
    - Iterate over each combination of parameter values in `params` using nested loops, ensuring the order minimizes model reloads.
    - For each combination, check if `n_prompt`, `n_gen`, or `n_pg` values are non-zero to create a `cmd_params_instance`.
    - If `n_prompt` is non-zero, create an instance with `n_prompt` set and `n_gen` as 0, then add it to `instances`.
    - If `n_gen` is non-zero, create an instance with `n_gen` set and `n_prompt` as 0, then add it to `instances`.
    - If `n_pg` has non-zero values, create an instance with both `n_prompt` and `n_gen` set from `n_pg`, then add it to `instances`.
    - Return the `instances` vector containing all generated `cmd_params_instance` objects.
- **Output**: A vector of `cmd_params_instance` objects, each representing a unique combination of parameter values from the input `cmd_params`.


---
### escape\_json<!-- {{#callable:escape_json}} -->
The `escape_json` function takes a string and returns a new string with special JSON characters escaped.
- **Inputs**:
    - `value`: A constant reference to a `std::string` that represents the input string to be escaped.
- **Control Flow**:
    - Initialize an empty string `escaped` to store the result.
    - Iterate over each character `c` in the input string `value`.
    - If `c` is a double quote (`"`), append `\"` to `escaped`.
    - If `c` is a backslash (`\`), append `\\` to `escaped`.
    - If `c` is a control character (ASCII value <= 0x1f), format it as a Unicode escape sequence `\uXXXX` and append to `escaped`.
    - Otherwise, append the character `c` to `escaped`.
- **Output**: Returns a `std::string` containing the escaped version of the input string.


---
### format\_json\_value<!-- {{#callable:format_json_value}} -->
The `format_json_value` function formats a given value as a JSON-compatible string based on the field type.
- **Inputs**:
    - `field`: A string representing the name of the field whose type is to be determined.
    - `value`: A string representing the value to be formatted according to the field type.
- **Control Flow**:
    - The function retrieves the field type using `test::get_field_type(field)`.
    - It uses a switch statement to handle different field types: `STRING`, `BOOL`, and a default case.
    - For `STRING` type, it returns the value enclosed in double quotes after escaping JSON special characters using [`escape_json`](#escape_json).
    - For `BOOL` type, it returns "false" if the value is "0", otherwise "true".
    - For other types, it returns the value as is.
- **Output**: A string formatted as a JSON-compatible value based on the field type.
- **Functions called**:
    - [`escape_json`](#escape_json)


---
### test\_prompt<!-- {{#callable:test_prompt}} -->
The `test_prompt` function tests the prompt processing capabilities of a llama model by decoding batches of tokens until a specified number of prompts have been processed.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` object, which represents the context in which the llama model operates.
    - `n_prompt`: An integer specifying the total number of prompts to process.
    - `n_batch`: An integer specifying the number of tokens to process in each batch.
    - `n_threads`: An integer specifying the number of threads to use for processing.
- **Control Flow**:
    - Set the number of threads for the llama context using `llama_set_n_threads`.
    - Retrieve the model and vocabulary from the context.
    - Initialize a vector of tokens with size `n_batch`.
    - Initialize `n_processed` to 0 to track the number of processed prompts.
    - Enter a loop that continues until `n_processed` is less than `n_prompt`.
    - Calculate `n_tokens` as the minimum of remaining prompts and `n_batch`.
    - Set the first token in the batch to the beginning-of-sequence token if applicable, otherwise a random token from the vocabulary.
    - Fill the rest of the batch with random tokens from the vocabulary.
    - Decode the batch of tokens using `llama_decode` and check for errors.
    - If decoding fails, print an error message and return false.
    - Increment `n_processed` by `n_tokens`.
    - After processing all prompts, synchronize the context with `llama_synchronize`.
    - Return true to indicate successful processing.
- **Output**: A boolean value indicating whether the prompt processing was successful (true) or not (false).


---
### test\_gen<!-- {{#callable:test_gen}} -->
The `test_gen` function tests the generation capabilities of a llama model by generating a specified number of tokens using a given number of threads.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure, which represents the context of the llama model being used.
    - `n_gen`: An integer specifying the number of tokens to generate.
    - `n_threads`: An integer specifying the number of threads to use for the generation process.
- **Control Flow**:
    - Set the number of threads for the llama context using `llama_set_n_threads` with the provided `n_threads` value.
    - Retrieve the model and vocabulary from the llama context using `llama_get_model` and `llama_model_get_vocab`, respectively.
    - Determine the number of tokens in the vocabulary using `llama_vocab_n_tokens`.
    - Initialize a token to either the beginning-of-sequence token or a random token from the vocabulary, depending on whether the vocabulary includes a beginning-of-sequence token.
    - Iterate `n_gen` times to generate tokens:
    - In each iteration, decode a single token using `llama_decode` and check for errors.
    - If decoding fails, print an error message and return `false`.
    - Synchronize the llama context using `llama_synchronize`.
    - Select a new random token from the vocabulary for the next iteration.
    - If all tokens are generated successfully, return `true`.
- **Output**: A boolean value indicating whether the token generation was successful (`true`) or not (`false`).


---
### llama\_null\_log\_callback<!-- {{#callable:llama_null_log_callback}} -->
The `llama_null_log_callback` function is a no-operation logging callback that ignores its input parameters.
- **Inputs**:
    - `level`: An enumeration value of type `ggml_log_level` representing the log level.
    - `text`: A constant character pointer to the log message text.
    - `user_data`: A void pointer to user-defined data, which is not used in this function.
- **Control Flow**:
    - The function takes three parameters: `level`, `text`, and `user_data`.
    - Each parameter is explicitly cast to void to indicate they are unused.
    - The function body contains no other logic or operations.
- **Output**: The function does not produce any output or return a value.


---
### create\_printer<!-- {{#callable:create_printer}} -->
The `create_printer` function creates and returns a unique pointer to a printer object based on the specified output format.
- **Inputs**:
    - `format`: An `output_formats` enum value that specifies the desired output format for the printer, such as CSV, JSON, JSONL, MARKDOWN, or SQL.
- **Control Flow**:
    - The function uses a switch statement to determine the type of printer to create based on the `format` argument.
    - If the format is `NONE`, the function returns a `nullptr`.
    - For each specific format (CSV, JSON, JSONL, MARKDOWN, SQL), the function creates a new printer object of the corresponding type and returns it as a unique pointer.
    - If the format does not match any of the specified cases, the function calls `GGML_ABORT` with a "fatal error" message.
- **Output**: A `std::unique_ptr<printer>` pointing to the newly created printer object, or `nullptr` if the format is `NONE`.


---
### main<!-- {{#callable:main}} -->
The `main` function initializes and executes a benchmarking process for a machine learning model using various configurations and outputs the results in specified formats.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of character pointers listing all the arguments.
- **Control Flow**:
    - Set the locale to UTF-8 for handling Unicode characters.
    - Print warnings if the build is in debug mode or if sanitizers are enabled, which may affect performance.
    - Initialize all backends using `ggml_backend_load_all()`.
    - Parse command-line parameters using `parse_cmd_params(argc, argv)`.
    - Check if the CPU backend is loaded and retrieve function pointers for threadpool creation and destruction.
    - Initialize the llama backend and NUMA settings based on parsed parameters.
    - Set the process priority according to the parsed parameters.
    - Create output printers for standard output and error based on specified formats.
    - Iterate over command parameter instances to perform benchmarks.
    - For each instance, load the model if necessary and create a context for it.
    - Perform warmup runs for prompt and generation if specified.
    - Execute the benchmark tests for the specified number of repetitions, measuring time and storing results.
    - Print the test results using the configured printers.
    - Free resources such as contexts, models, and threadpools after each test.
    - Print footers for the output formats and free the llama backend resources.
    - Return 0 to indicate successful execution.
- **Output**: The function returns an integer, 0 for successful execution or 1 if an error occurs during initialization or execution.
- **Functions called**:
    - [`ggml_backend_load_all`](../../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_load_all)
    - [`parse_cmd_params`](#parse_cmd_params)
    - [`ggml_backend_reg_get_proc_address`](../../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address)
    - [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init)
    - [`llama_numa_init`](../../src/llama.cpp.driver.md#llama_numa_init)
    - [`create_printer`](#create_printer)
    - [`get_cmd_params_instances`](#get_cmd_params_instances)
    - [`ggml_threadpool_params_default`](../../ggml/src/ggml.c.driver.md#ggml_threadpool_params_default)
    - [`llama_attach_threadpool`](../../src/llama-context.cpp.driver.md#llama_attach_threadpool)
    - [`test_prompt`](#test_prompt)
    - [`test_gen`](#test_gen)
    - [`get_time_ns`](#get_time_ns)
    - [`llama_backend_free`](../../src/llama.cpp.driver.md#llama_backend_free)


