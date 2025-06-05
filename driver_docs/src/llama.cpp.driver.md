# Purpose
This C++ source code file is part of a larger software system related to the "llama" project, which appears to be focused on model management and processing, likely in the context of machine learning or artificial intelligence. The file provides a range of functionalities, including model loading, saving, and device management, as well as support for various backend operations. It includes several key components such as [`llama_model_load`](#llama_model_load), which handles the loading of models from files, and [`llama_model_save_to_file`](#llama_model_save_to_file), which manages saving models to files. The code also includes functions for initializing and freeing backend resources, checking system capabilities like memory mapping and GPU offloading, and handling NUMA (Non-Uniform Memory Access) strategies.

The file is not an executable but rather a C++ source file intended to be part of a larger library or application. It defines several public APIs and interfaces, such as [`llama_model_load_from_file`](#llama_model_load_from_file) and [`llama_model_save_to_file`](#llama_model_save_to_file), which are likely intended to be used by other parts of the software or by external applications. The code also includes deprecated functions, indicating ongoing development and updates. Additionally, the file manages device configurations and supports operations across different hardware backends, which is crucial for optimizing performance in computationally intensive tasks. Overall, this file provides a comprehensive set of functionalities for managing and interacting with models within the "llama" system.
# Imports and Dependencies

---
- `llama-impl.h`
- `llama-chat.h`
- `llama-mmap.h`
- `llama-vocab.h`
- `llama-model-loader.h`
- `llama-model-saver.h`
- `llama-model.h`
- `ggml.h`
- `ggml-backend.h`
- `algorithm`
- `cstddef`
- `cstdint`
- `cstdio`
- `cstring`
- `ctime`


# Functions

---
### llama\_sampler\_chain\_default\_params<!-- {{#callable:llama_sampler_chain_default_params}} -->
The `llama_sampler_chain_default_params` function initializes and returns a `llama_sampler_chain_params` structure with default values.
- **Inputs**: None
- **Control Flow**:
    - A `llama_sampler_chain_params` structure named `result` is initialized with the field `no_perf` set to `true`.
    - The function returns the `result` structure.
- **Output**: A `llama_sampler_chain_params` structure with default values.


---
### llama\_max\_devices<!-- {{#callable:llama_max_devices}} -->
The `llama_max_devices` function returns the maximum number of devices supported, which is 16.
- **Inputs**: None
- **Control Flow**:
    - The function is defined to return a constant value.
    - It directly returns the integer value 16 without any conditions or calculations.
- **Output**: The function returns a `size_t` integer value representing the maximum number of devices, which is 16.


---
### llama\_supports\_mmap<!-- {{#callable:llama_supports_mmap}} -->
The function `llama_supports_mmap` checks if memory-mapped file support is available in the llama library.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of `llama_mmap::SUPPORTED`.
- **Output**: A boolean value indicating whether memory-mapped file support is available.


---
### llama\_supports\_mlock<!-- {{#callable:llama_supports_mlock}} -->
The `llama_supports_mlock` function checks if the mlock feature is supported by returning a boolean value.
- **Inputs**: None
- **Control Flow**:
    - The function directly returns the value of `llama_mlock::SUPPORTED`.
- **Output**: A boolean value indicating whether mlock is supported.


---
### llama\_supports\_gpu\_offload<!-- {{#callable:llama_supports_gpu_offload}} -->
The function `llama_supports_gpu_offload` checks if GPU offloading is supported by verifying the availability of a GPU backend or RPC support.
- **Inputs**: None
- **Control Flow**:
    - The function calls `ggml_backend_dev_by_type` with `GGML_BACKEND_DEVICE_TYPE_GPU` to check if a GPU backend is available.
    - If a GPU backend is available, the function returns `true`.
    - If no GPU backend is available, the function calls [`llama_supports_rpc`](#llama_supports_rpc) to check for RPC support.
    - If RPC support is available, the function returns `true`.
    - If neither a GPU backend nor RPC support is available, the function returns `false`.
- **Output**: A boolean value indicating whether GPU offloading is supported.
- **Functions called**:
    - [`llama_supports_rpc`](#llama_supports_rpc)


---
### llama\_supports\_rpc<!-- {{#callable:llama_supports_rpc}} -->
The `llama_supports_rpc` function checks if the 'RPC' backend is registered and available.
- **Inputs**: None
- **Control Flow**:
    - The function calls `ggml_backend_reg_by_name` with the argument 'RPC'.
    - It checks if the return value of `ggml_backend_reg_by_name` is not `nullptr`.
    - If the return value is not `nullptr`, it returns `true`, indicating that the 'RPC' backend is supported.
    - If the return value is `nullptr`, it returns `false`, indicating that the 'RPC' backend is not supported.
- **Output**: A boolean value indicating whether the 'RPC' backend is supported (`true`) or not (`false`).


---
### llama\_backend\_init<!-- {{#callable:llama_backend_init}} -->
The `llama_backend_init` function initializes the backend by setting up timing and preparing f16 tables through a temporary ggml context.
- **Inputs**: None
- **Control Flow**:
    - Call `ggml_time_init()` to initialize timing functions.
    - Create a `ggml_init_params` structure with default values (zero size, null pointer, and false flag).
    - Initialize a `ggml_context` using `ggml_init()` with the created parameters.
    - Free the `ggml_context` using `ggml_free()` to clean up resources.
- **Output**: The function does not return any value (void).
- **Functions called**:
    - [`ggml_init`](../ggml/src/ggml.c.driver.md#ggml_init)
    - [`ggml_free`](../ggml/src/ggml.c.driver.md#ggml_free)


---
### llama\_numa\_init<!-- {{#callable:llama_numa_init}} -->
The `llama_numa_init` function initializes NUMA (Non-Uniform Memory Access) settings for the CPU backend if NUMA is not disabled.
- **Inputs**:
    - `numa`: An enumeration value of type `ggml_numa_strategy` that specifies the NUMA strategy to be used.
- **Control Flow**:
    - Check if the `numa` strategy is not `GGML_NUMA_STRATEGY_DISABLED`.
    - Retrieve the CPU backend device using `ggml_backend_dev_by_type` with `GGML_BACKEND_DEVICE_TYPE_CPU`.
    - Assert that the CPU backend device is loaded.
    - Get the backend registry for the CPU device using `ggml_backend_dev_backend_reg`.
    - Retrieve the function pointer for `ggml_backend_cpu_numa_init` from the registry using [`ggml_backend_reg_get_proc_address`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address).
    - Call the retrieved NUMA initialization function with the `numa` strategy.
- **Output**: The function does not return any value.
- **Functions called**:
    - [`ggml_backend_reg_get_proc_address`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address)


---
### llama\_backend\_free<!-- {{#callable:llama_backend_free}} -->
The `llama_backend_free` function releases resources allocated for quantization in the backend.
- **Inputs**: None
- **Control Flow**:
    - The function calls `ggml_quantize_free()` to free resources related to quantization.
- **Output**: The function does not return any value.
- **Functions called**:
    - [`ggml_quantize_free`](../ggml/src/ggml.c.driver.md#ggml_quantize_free)


---
### llama\_time\_us<!-- {{#callable:llama_time_us}} -->
The `llama_time_us` function returns the current time in microseconds by calling the `ggml_time_us` function.
- **Inputs**: None
- **Control Flow**:
    - The function directly calls the `ggml_time_us` function.
    - It returns the result of the `ggml_time_us` function call.
- **Output**: The function returns an `int64_t` value representing the current time in microseconds.


---
### llama\_model\_load<!-- {{#callable:llama_model_load}} -->
The `llama_model_load` function loads a machine learning model from a file, handling various components like architecture, hyperparameters, vocabulary, and tensors, while managing errors and logging progress.
- **Inputs**:
    - `fname`: A string representing the filename of the model to be loaded.
    - `splits`: A vector of strings that may contain split information for the model file.
    - `model`: A reference to a `llama_model` object where the loaded model data will be stored.
    - `params`: A reference to a `llama_model_params` object containing parameters that influence the loading process, such as whether to use memory mapping or load only the vocabulary.
- **Control Flow**:
    - Initialize the model's loading time and start time using a `time_meas` object.
    - Create a `llama_model_loader` object with the provided filename, splits, and parameters.
    - Print model loader information.
    - Set the model's vocabulary-only flag based on the parameters.
    - Attempt to load the model architecture, catching and throwing an error if it fails.
    - Attempt to load the model hyperparameters, catching and throwing an error if it fails.
    - Attempt to load the model vocabulary, catching and throwing an error if it fails.
    - Load model statistics and print model information.
    - If the `vocab_only` parameter is true, log a message and return 0, skipping tensor loading.
    - Attempt to load the model tensors; if unsuccessful, return -2.
    - Catch any exceptions during the loading process, log an error message, and return -1.
- **Output**: Returns 0 on successful loading, -1 if an error occurs, and -2 if tensor loading is cancelled.


---
### llama\_model\_load\_from\_file\_impl<!-- {{#callable:llama_model_load_from_file_impl}} -->
The `llama_model_load_from_file_impl` function initializes and loads a llama model from a specified file path, configuring device usage and handling progress callbacks.
- **Inputs**:
    - `path_model`: A string representing the file path to the model to be loaded.
    - `splits`: A vector of strings that may contain split file paths for the model.
    - `params`: A `llama_model_params` structure containing parameters for model loading, such as device configuration and progress callback.
- **Control Flow**:
    - Initialize the timing system with `ggml_time_init()`.
    - Check if backends are loaded if `vocab_only` is false; log an error and return `nullptr` if not.
    - Set up a default progress callback if none is provided, updating a percentage and logging progress.
    - Create a new `llama_model` instance with the provided parameters.
    - Determine the devices to use for the model based on `params.devices` or available devices, prioritizing RPC servers.
    - If in single GPU mode, validate and retain only the main GPU, logging an error and returning `nullptr` if invalid.
    - Log memory information for each device being used.
    - Call [`llama_model_load`](#llama_model_load) to load the model from the file, checking the status for errors or cancellation.
    - Free the model and return `nullptr` if loading fails or is cancelled.
    - Return the loaded `llama_model` instance if successful.
- **Output**: A pointer to a `llama_model` structure if the model is successfully loaded, or `nullptr` if an error occurs.
- **Functions called**:
    - [`ggml_backend_reg_count`](../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_reg_count)
    - [`ggml_backend_dev_count`](../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_dev_count)
    - [`ggml_backend_dev_type`](../ggml/include/ggml-backend.h.driver.md#ggml_backend_dev_type)
    - [`ggml_backend_reg_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_name)
    - [`ggml_backend_dev_memory`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_memory)
    - [`ggml_backend_dev_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_name)
    - [`ggml_backend_dev_description`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_description)
    - [`llama_model_load`](#llama_model_load)


---
### llama\_load\_model\_from\_file<!-- {{#callable:llama_load_model_from_file}} -->
The `llama_load_model_from_file` function loads a llama model from a specified file path using given model parameters.
- **Inputs**:
    - `path_model`: A constant character pointer representing the file path to the model to be loaded.
    - `params`: A structure of type `llama_model_params` containing parameters for loading the model.
- **Control Flow**:
    - The function directly calls [`llama_model_load_from_file`](#llama_model_load_from_file) with the provided `path_model` and `params` arguments.
    - The [`llama_model_load_from_file`](#llama_model_load_from_file) function initializes an empty vector for splits and calls `llama_model_load_from_file_impl` with the path, splits, and parameters.
    - The `llama_model_load_from_file_impl` function handles the actual loading process, including backend checks, device setup, and model loading using `llama_model_load`.
- **Output**: Returns a pointer to a `llama_model` structure representing the loaded model, or `nullptr` if loading fails.
- **Functions called**:
    - [`llama_model_load_from_file`](#llama_model_load_from_file)


---
### llama\_model\_load\_from\_file<!-- {{#callable:llama_model_load_from_file}} -->
The `llama_model_load_from_file` function loads a llama model from a specified file path using given parameters.
- **Inputs**:
    - `path_model`: A constant character pointer representing the file path to the model to be loaded.
    - `params`: A structure of type `llama_model_params` containing parameters for loading the model.
- **Control Flow**:
    - Initialize an empty vector of strings named `splits`.
    - Call the [`llama_model_load_from_file_impl`](#llama_model_load_from_file_impl) function with `path_model`, `splits`, and `params` as arguments.
    - Return the result of the [`llama_model_load_from_file_impl`](#llama_model_load_from_file_impl) function call.
- **Output**: Returns a pointer to a `llama_model` structure if successful, or `nullptr` if the model loading fails.
- **Functions called**:
    - [`llama_model_load_from_file_impl`](#llama_model_load_from_file_impl)


---
### llama\_model\_load\_from\_splits<!-- {{#callable:llama_model_load_from_splits}} -->
The `llama_model_load_from_splits` function loads a llama model from multiple file paths specified as splits.
- **Inputs**:
    - `paths`: An array of C-style strings representing the file paths to the model splits.
    - `n_paths`: The number of paths provided in the `paths` array.
    - `params`: A `llama_model_params` structure containing parameters for loading the model.
- **Control Flow**:
    - Initialize an empty vector `splits` to store the paths.
    - Check if `n_paths` is zero; if so, log an error and return `nullptr`.
    - Iterate over the `paths` array and add each path to the `splits` vector.
    - Call [`llama_model_load_from_file_impl`](#llama_model_load_from_file_impl) with the first path in `splits`, the `splits` vector, and `params`, and return its result.
- **Output**: Returns a pointer to a `llama_model` structure if successful, or `nullptr` if an error occurs.
- **Functions called**:
    - [`llama_model_load_from_file_impl`](#llama_model_load_from_file_impl)


---
### llama\_model\_save\_to\_file<!-- {{#callable:llama_model_save_to_file}} -->
The `llama_model_save_to_file` function saves a given llama model to a specified file path.
- **Inputs**:
    - `model`: A pointer to a `llama_model` structure representing the model to be saved.
    - `path_model`: A constant character pointer representing the file path where the model should be saved.
- **Control Flow**:
    - Create an instance of `llama_model_saver` using the provided model.
    - Call `add_kv_from_model` on the `llama_model_saver` instance to add key-value pairs from the model.
    - Call `add_tensors_from_model` on the `llama_model_saver` instance to add tensors from the model.
    - Call `save` on the `llama_model_saver` instance with `path_model` to save the model to the specified file path.
- **Output**: The function does not return any value; it performs the side effect of saving the model to a file.


---
### llama\_chat\_apply\_template<!-- {{#callable:llama_chat_apply_template}} -->
The `llama_chat_apply_template` function formats a series of chat messages using a specified template and stores the result in a buffer.
- **Inputs**:
    - `tmpl`: A pointer to a character string representing the template to be used for formatting; if null, defaults to "chatml".
    - `chat`: A pointer to an array of `llama_chat_message` structures representing the chat messages to be formatted.
    - `n_msg`: The number of chat messages in the `chat` array.
    - `add_ass`: A boolean flag indicating whether to add additional assertions during formatting.
    - `buf`: A pointer to a character buffer where the formatted chat string will be stored.
    - `length`: The maximum length of the buffer `buf`.
- **Control Flow**:
    - Initialize `curr_tmpl` with the provided template or default to "chatml" if null.
    - Create a vector `chat_vec` to store pointers to the chat messages.
    - Detect the chat template using [`llm_chat_detect_template`](llama-chat.cpp.driver.md#llm_chat_detect_template) with `curr_tmpl`.
    - If the detected template is unknown, return -1 indicating an error.
    - Apply the detected template to the chat messages using [`llm_chat_apply_template`](llama-chat.cpp.driver.md#llm_chat_apply_template), storing the result in `formatted_chat`.
    - If the result of applying the template is negative, return the result as an error code.
    - If a buffer `buf` is provided and its length is greater than zero, copy the formatted chat string into `buf` using `strncpy`.
    - Return the result of the template application.
- **Output**: Returns an integer indicating the success or failure of the operation: 0 on success, -1 if the template is unknown, or a negative value from [`llm_chat_apply_template`](llama-chat.cpp.driver.md#llm_chat_apply_template) if an error occurs during formatting.
- **Functions called**:
    - [`llm_chat_detect_template`](llama-chat.cpp.driver.md#llm_chat_detect_template)
    - [`llm_chat_apply_template`](llama-chat.cpp.driver.md#llm_chat_apply_template)


---
### llama\_split\_path<!-- {{#callable:llama_split_path}} -->
The `llama_split_path` function formats a file path string for a specific split of a dataset using a given prefix and split indices.
- **Inputs**:
    - `split_path`: A character array where the formatted split path will be stored.
    - `maxlen`: The maximum length of the split_path buffer.
    - `path_prefix`: A string representing the prefix for the path.
    - `split_no`: An integer representing the current split number (0-based index).
    - `split_count`: An integer representing the total number of splits.
- **Control Flow**:
    - Define a static format string `SPLIT_PATH_FORMAT` for the split path.
    - Use `snprintf` to format the split path using the provided prefix, split number (incremented by 1 for 1-based index), and total split count.
    - If `snprintf` successfully writes to `split_path`, return the length of the resulting string.
    - If `snprintf` fails, return 0.
- **Output**: The function returns the length of the formatted split path string if successful, otherwise it returns 0.


---
### llama\_split\_prefix<!-- {{#callable:llama_split_prefix}} -->
The `llama_split_prefix` function checks if a given file path ends with a specific postfix pattern and extracts the prefix if it does.
- **Inputs**:
    - `split_prefix`: A character array where the extracted prefix will be stored if the condition is met.
    - `maxlen`: The maximum length of the `split_prefix` buffer.
    - `split_path`: The full file path as a constant character string to be checked against the postfix pattern.
    - `split_no`: An integer representing the current split number, used to construct the postfix.
    - `split_count`: An integer representing the total number of splits, used to construct the postfix.
- **Control Flow**:
    - Convert `split_path` to a `std::string` for easier manipulation.
    - Construct a postfix string using `split_no` and `split_count` in the format '-%05d-of-%05d.gguf'.
    - Calculate the size of the prefix by subtracting the size of the postfix from the size of `split_path`.
    - Check if the calculated prefix size is positive and if `split_path` ends with the constructed postfix.
    - If both conditions are met, copy the prefix part of `split_path` into `split_prefix` and return the size of the prefix.
    - If the conditions are not met, return 0.
- **Output**: Returns the size of the prefix if the conditions are met, otherwise returns 0.


---
### llama\_print\_system\_info<!-- {{#callable:llama_print_system_info}} -->
The `llama_print_system_info` function retrieves and formats system information from registered backends into a string.
- **Inputs**: None
- **Control Flow**:
    - Initialize a static string `s` and clear its contents to avoid data accumulation from previous calls.
    - Iterate over each registered backend using `ggml_backend_reg_count()` to determine the number of backends.
    - For each backend, retrieve the backend registration using `ggml_backend_reg_get(i)`.
    - Attempt to get the function pointer for `ggml_backend_get_features` using [`ggml_backend_reg_get_proc_address`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address).
    - If the function pointer is valid, call it to retrieve the features of the backend.
    - Append the backend's name and its features (name and value) to the string `s`, formatting them with separators.
- **Output**: A C-style string containing the formatted system information of all registered backends.
- **Functions called**:
    - [`ggml_backend_reg_count`](../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_reg_count)
    - [`ggml_backend_reg_get_proc_address`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address)
    - [`ggml_backend_reg_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_name)


