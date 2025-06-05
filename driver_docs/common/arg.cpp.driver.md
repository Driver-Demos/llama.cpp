# Purpose
This C++ source code file is a comprehensive utility for handling command-line arguments and configurations for a software application, likely related to machine learning or data processing. The file includes a variety of functionalities, such as reading and writing files, managing JSON data, handling environment variables, and supporting network operations with optional CURL integration. It defines a set of command-line options and parameters, which are used to configure the behavior of the application, including model paths, sampling parameters, and system settings like CPU affinity and thread management.

The code is structured to support a wide range of functionalities, including speculative decoding, embedding, and server operations. It provides mechanisms for downloading models from remote URLs, handling multimodal data, and managing GPU resources. The file also includes utilities for parsing and validating command-line arguments, with support for environment variables and default values. Additionally, it offers features for logging, performance monitoring, and error handling, making it a robust foundation for applications that require flexible configuration and resource management. The presence of numerous conditional compilation directives and platform-specific code indicates that the file is designed to be portable across different operating systems.
# Imports and Dependencies

---
- `arg.h`
- `chat.h`
- `common.h`
- `gguf.h`
- `json-schema-to-grammar.h`
- `log.h`
- `sampling.h`
- `windows.h`
- `nlohmann/json.hpp`
- `algorithm`
- `climits`
- `cstdarg`
- `filesystem`
- `fstream`
- `regex`
- `set`
- `string`
- `thread`
- `vector`
- `curl/curl.h`
- `curl/easy.h`
- `future`
- `linux/limits.h`
- `sys/limits.h`
- `sys/syslimits.h`


# Global Variables

---
### mmproj\_examples
- **Type**: `std::initializer_list<enum llama_example>`
- **Description**: The `mmproj_examples` variable is a global variable defined as an `std::initializer_list` containing elements of the `llama_example` enumeration type. It is initialized with two specific enumeration values: `LLAMA_EXAMPLE_MTMD` and `LLAMA_EXAMPLE_SERVER`. This list is used to specify a set of examples related to the 'mmproj' functionality.
- **Use**: This variable is used to define a list of examples that are relevant for multimodal projection (mmproj) operations.


---
### kv\_cache\_types
- **Type**: `std::vector<ggml_type>`
- **Description**: The `kv_cache_types` is a global constant vector that holds a list of `ggml_type` enumerations. These types represent different data formats that can be used for key-value caching in the application.
- **Use**: This variable is used to define the supported data types for key-value cache operations, allowing the application to handle various precision and quantization formats.


# Data Structures

---
### common\_hf\_file\_res<!-- {{#data_structure:common_hf_file_res}} -->
- **Type**: `struct`
- **Members**:
    - `repo`: A string representing the repository name with the ":tag" removed.
    - `ggufFile`: A string representing the GGUF file name.
    - `mmprojFile`: A string representing the MMProj file name.
- **Description**: The `common_hf_file_res` struct is a simple data structure used to store information about a repository and associated files. It contains three string members: `repo`, which holds the repository name with any ":tag" suffix removed; `ggufFile`, which stores the name of a GGUF file; and `mmprojFile`, which stores the name of an MMProj file. This struct is likely used to manage or reference files related to a specific repository in a larger system.


---
### curl\_slist\_ptr<!-- {{#data_structure:curl_slist_ptr}} -->
- **Type**: `struct`
- **Members**:
    - `ptr`: A pointer to a `curl_slist` structure, initialized to `nullptr`.
- **Description**: The `curl_slist_ptr` is a C++ struct designed to manage a pointer to a `curl_slist` structure, which is used in the libcurl library to handle linked lists of strings. The struct includes a destructor that automatically frees the memory associated with the `curl_slist` when the `curl_slist_ptr` object is destroyed, ensuring proper resource management and preventing memory leaks.
- **Member Functions**:
    - [`curl_slist_ptr::~curl_slist_ptr`](#curl_slist_ptrcurl_slist_ptr)

**Methods**

---
#### curl\_slist\_ptr::\~curl\_slist\_ptr<!-- {{#callable:curl_slist_ptr::~curl_slist_ptr}} -->
The destructor `~curl_slist_ptr` releases memory allocated for a `curl_slist` linked list if it exists.
- **Inputs**: None
- **Control Flow**:
    - Check if the `ptr` member is not null.
    - If `ptr` is not null, call `curl_slist_free_all(ptr)` to free the memory allocated for the `curl_slist`.
- **Output**: This destructor does not return any value; it ensures that the memory for the `curl_slist` is properly freed when a `curl_slist_ptr` object is destroyed.
- **See also**: [`curl_slist_ptr`](#curl_slist_ptr)  (Data Structure)



---
### common\_load\_model\_from\_url\_headers<!-- {{#data_structure:common_download_file_single::common_load_model_from_url_headers}} -->
- **Type**: `struct`
- **Members**:
    - `etag`: A string representing the entity tag (ETag) of a resource.
    - `last_modified`: A string representing the last modified date of a resource.
- **Description**: The `common_load_model_from_url_headers` struct is a simple data structure used to store HTTP header information related to a resource fetched from a URL. It contains two string members: `etag`, which holds the entity tag of the resource, and `last_modified`, which stores the last modified date of the resource. This struct is typically used in the context of HTTP requests to manage caching and resource validation by comparing these headers with those of a previously fetched resource.


---
### FILE\_deleter<!-- {{#data_structure:common_download_file_single::FILE_deleter}} -->
- **Type**: `struct`
- **Description**: The `FILE_deleter` struct is a custom deleter for `FILE` pointers, designed to be used with smart pointers like `std::unique_ptr`. It provides an `operator()` that takes a `FILE*` and calls `fclose()` on it, ensuring that the file is properly closed when the smart pointer goes out of scope.
- **Member Functions**:
    - [`common_download_file_single::FILE_deleter::operator()`](#FILE_deleteroperator())

**Methods**

---
#### FILE\_deleter::operator\(\)<!-- {{#callable:common_download_file_single::FILE_deleter::operator()}} -->
The `operator()` function in the `FILE_deleter` struct is used to close a file pointer using `fclose`.
- **Inputs**:
    - `f`: A pointer to a FILE object that needs to be closed.
- **Control Flow**:
    - The function takes a FILE pointer as an argument.
    - It calls the `fclose` function on the provided FILE pointer to close the file.
- **Output**: The function does not return any value.
- **See also**: [`common_download_file_single::FILE_deleter`](#common_download_file_singleFILE_deleter)  (Data Structure)



---
### handle\_model\_result<!-- {{#data_structure:handle_model_result}} -->
- **Type**: `struct`
- **Members**:
    - `found_mmproj`: Indicates whether a multimodal projector was found.
    - `mmproj`: Holds the common parameters for the multimodal projector model.
- **Description**: The `handle_model_result` struct is designed to encapsulate the result of handling a model, specifically focusing on multimodal projector models. It contains a boolean flag `found_mmproj` to indicate if a multimodal projector was found, and a `common_params_model` object `mmproj` to store the parameters related to the multimodal projector model. This struct is useful for managing and passing around the state and configuration of multimodal projector models within the application.


---
### common\_arg<!-- {{#data_structure:common_arg}} -->
- **Description**: [See definition](arg.h.driver.md#common_arg)
- **Member Functions**:
    - [`common_arg::set_examples`](#common_argset_examples)
    - [`common_arg::set_excludes`](#common_argset_excludes)
    - [`common_arg::set_env`](#common_argset_env)
    - [`common_arg::set_sparam`](#common_argset_sparam)
    - [`common_arg::in_example`](#common_argin_example)
    - [`common_arg::is_exclude`](#common_argis_exclude)
    - [`common_arg::get_value_from_env`](#common_argget_value_from_env)
    - [`common_arg::has_value_from_env`](#common_arghas_value_from_env)
    - [`common_arg::to_string`](#common_argto_string)
    - [`common_arg::common_arg`](arg.h.driver.md#common_argcommon_arg)
    - [`common_arg::common_arg`](arg.h.driver.md#common_argcommon_arg)
    - [`common_arg::common_arg`](arg.h.driver.md#common_argcommon_arg)
    - [`common_arg::common_arg`](arg.h.driver.md#common_argcommon_arg)

**Methods**

---
#### common\_arg::set\_examples<!-- {{#callable:common_arg::set_examples}} -->
The `set_examples` function assigns a new set of `llama_example` enums to the `examples` member of a `common_arg` object.
- **Inputs**:
    - `examples`: An `std::initializer_list` of `llama_example` enums that will be used to update the `examples` member of the `common_arg` object.
- **Control Flow**:
    - The function takes an `initializer_list` of `llama_example` enums as input.
    - It uses `std::move` to transfer the contents of the input list to the `examples` member of the `common_arg` object.
    - The function returns a reference to the current `common_arg` object (`*this`).
- **Output**: A reference to the modified `common_arg` object, allowing for method chaining.
- **See also**: [`common_arg`](arg.h.driver.md#common_arg)  (Data Structure)


---
#### common\_arg::set\_excludes<!-- {{#callable:common_arg::set_excludes}} -->
The `set_excludes` method assigns a new set of `llama_example` enums to the `excludes` member of the `common_arg` structure.
- **Inputs**:
    - `excludes`: An initializer list of `llama_example` enums that will be assigned to the `excludes` member of the `common_arg` structure.
- **Control Flow**:
    - The method takes an initializer list of `llama_example` enums as input.
    - It assigns this list to the `excludes` member of the `common_arg` structure using `std::move` to efficiently transfer ownership.
    - The method returns a reference to the current `common_arg` object, allowing for method chaining.
- **Output**: A reference to the current `common_arg` object, allowing for method chaining.
- **See also**: [`common_arg`](arg.h.driver.md#common_arg)  (Data Structure)


---
#### common\_arg::set\_env<!-- {{#callable:common_arg::set_env}} -->
The `set_env` method sets the environment variable for a `common_arg` object and updates its help string to include this environment information.
- **Inputs**:
    - `env`: A constant character pointer representing the environment variable to be set for the `common_arg` object.
- **Control Flow**:
    - The method appends the environment variable information to the `help` string of the `common_arg` object.
    - The method assigns the provided `env` value to the `env` member variable of the `common_arg` object.
    - The method returns a reference to the current `common_arg` object, allowing for method chaining.
- **Output**: A reference to the modified `common_arg` object, allowing for method chaining.
- **See also**: [`common_arg`](arg.h.driver.md#common_arg)  (Data Structure)


---
#### common\_arg::set\_sparam<!-- {{#callable:common_arg::set_sparam}} -->
The `set_sparam` method sets the `is_sparam` flag to true for a `common_arg` object and returns the modified object.
- **Inputs**: None
- **Control Flow**:
    - The method sets the `is_sparam` member variable to `true`.
    - The method returns the current instance of `common_arg` by dereferencing `this`.
- **Output**: The method returns a reference to the modified `common_arg` object.
- **See also**: [`common_arg`](arg.h.driver.md#common_arg)  (Data Structure)


---
#### common\_arg::in\_example<!-- {{#callable:common_arg::in_example}} -->
The `in_example` function checks if a given `llama_example` enum value is present in the `examples` set of a `common_arg` object.
- **Inputs**:
    - `ex`: An enum value of type `llama_example` that represents a specific example to check for in the `examples` set.
- **Control Flow**:
    - The function uses the `find` method of the `std::set` to search for the `ex` value in the `examples` set.
    - It compares the result of `find` with `examples.end()` to determine if `ex` is present in the set.
- **Output**: Returns `true` if the `ex` value is found in the `examples` set, otherwise returns `false`.
- **See also**: [`common_arg`](arg.h.driver.md#common_arg)  (Data Structure)


---
#### common\_arg::is\_exclude<!-- {{#callable:common_arg::is_exclude}} -->
The `is_exclude` function checks if a given `llama_example` is present in the `excludes` set of a `common_arg` object.
- **Inputs**:
    - `ex`: An enumeration value of type `llama_example` that represents a specific example to check against the `excludes` set.
- **Control Flow**:
    - The function checks if the provided `llama_example` (`ex`) is present in the `excludes` set of the `common_arg` object.
    - It uses the `find` method of the `std::set` to search for the `ex` value.
    - The function returns `true` if `ex` is found in the `excludes` set, otherwise it returns `false`.
- **Output**: A boolean value indicating whether the specified `llama_example` is in the `excludes` set.
- **See also**: [`common_arg`](arg.h.driver.md#common_arg)  (Data Structure)


---
#### common\_arg::get\_value\_from\_env<!-- {{#callable:common_arg::get_value_from_env}} -->
The `get_value_from_env` function retrieves the value of an environment variable specified by the `env` member of the `common_arg` structure and assigns it to the `output` string if it exists.
- **Inputs**:
    - `output`: A reference to a string where the environment variable's value will be stored if found.
- **Control Flow**:
    - Check if the `env` member is `nullptr`; if so, return `false`.
    - Use `std::getenv` to retrieve the value of the environment variable specified by `env`.
    - If the environment variable exists (i.e., `value` is not `nullptr`), assign it to `output` and return `true`.
    - If the environment variable does not exist, return `false`.
- **Output**: Returns a boolean indicating whether the environment variable was found and its value was successfully assigned to `output`.
- **See also**: [`common_arg`](arg.h.driver.md#common_arg)  (Data Structure)


---
#### common\_arg::has\_value\_from\_env<!-- {{#callable:common_arg::has_value_from_env}} -->
The `has_value_from_env` function checks if the `env` member of the `common_arg` structure is set and if the corresponding environment variable is defined.
- **Inputs**: None
- **Control Flow**:
    - Check if the `env` member is not `nullptr`.
    - Use `std::getenv` to check if the environment variable specified by `env` is set.
    - Return `true` if both conditions are met, otherwise return `false`.
- **Output**: A boolean value indicating whether the `env` member is set and the corresponding environment variable is defined.
- **See also**: [`common_arg`](arg.h.driver.md#common_arg)  (Data Structure)


---
#### common\_arg::to\_string<!-- {{#callable:common_arg::to_string}} -->
The `to_string` method of the `common_arg` class formats and returns a string representation of the command-line arguments, value hints, and help text associated with the `common_arg` instance.
- **Inputs**: None
- **Control Flow**:
    - Initialize constants for leading spaces and characters per line for help text.
    - Create a string of leading spaces for formatting.
    - Initialize a string stream to build the output string.
    - Iterate over each argument in the `args` vector.
    - For the first argument, check if it's the only one; if not, format it with padding and a comma.
    - For subsequent arguments, append them with a comma unless it's the last argument.
    - Append `value_hint` and `value_hint_2` to the string stream if they are not null.
    - Check if the current line length exceeds the allowed length; if so, add a newline and leading spaces.
    - Otherwise, add padding spaces to align the help text.
    - Break the `help` string into lines based on the maximum characters per line and append each line to the string stream, adding leading spaces for subsequent lines.
    - Return the constructed string from the string stream.
- **Output**: A formatted string representing the command-line arguments, value hints, and help text of the `common_arg` instance.
- **Functions called**:
    - [`break_str_into_lines`](#break_str_into_lines)
- **See also**: [`common_arg`](arg.h.driver.md#common_arg)  (Data Structure)



# Functions

---
### read\_file<!-- {{#callable:read_file}} -->
The `read_file` function reads the entire content of a file specified by its filename and returns it as a string.
- **Inputs**:
    - `fname`: A constant reference to a `std::string` representing the name of the file to be read.
- **Control Flow**:
    - Open the file specified by `fname` using an `std::ifstream` object.
    - Check if the file was successfully opened; if not, throw a `std::runtime_error` with an error message.
    - Read the entire content of the file into a `std::string` using `std::istreambuf_iterator`.
    - Close the file.
    - Return the content as a `std::string`.
- **Output**: A `std::string` containing the entire content of the file.


---
### write\_file<!-- {{#callable:write_file}} -->
The `write_file` function writes a given string content to a specified file, throwing an error if the file cannot be opened.
- **Inputs**:
    - `fname`: A constant reference to a `std::string` representing the name of the file to which the content will be written.
    - `content`: A constant reference to a `std::string` containing the content to be written to the file.
- **Control Flow**:
    - Open a file stream for writing using the provided file name `fname`.
    - Check if the file stream is successfully opened; if not, throw a `std::runtime_error` with an error message.
    - Write the `content` string to the file.
    - Close the file stream.
- **Output**: This function does not return any value; it performs its operations directly on the file system.


---
### break\_str\_into\_lines<!-- {{#callable:break_str_into_lines}} -->
The `break_str_into_lines` function splits a given input string into multiple lines, each with a maximum specified number of characters, while ensuring words are not split across lines.
- **Inputs**:
    - `input`: A string that needs to be split into lines.
    - `max_char_per_line`: The maximum number of characters allowed per line.
- **Control Flow**:
    - Initialize an empty vector `result` to store the resulting lines.
    - Create an input string stream `iss` from the input string.
    - Define a lambda function `add_line` to process each line, checking if it fits within the character limit or needs further splitting by words.
    - Iterate over each line in the input string using `std::getline`.
    - For each line, call the `add_line` lambda to process and add it to the `result` vector.
    - Return the `result` vector containing the split lines.
- **Output**: A vector of strings, where each string is a line with a maximum length of `max_char_per_line`.


---
### common\_has\_curl<!-- {{#callable:common_has_curl}} -->
The function `common_has_curl` checks if the program is built with CURL support and returns a boolean value accordingly.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the `LLAMA_USE_CURL` preprocessor directive is defined.
    - If `LLAMA_USE_CURL` is defined, the function returns `true`.
    - If `LLAMA_USE_CURL` is not defined, the function returns `false`.
- **Output**: A boolean value indicating whether CURL support is available (`true` if available, `false` otherwise).


---
### curl\_perform\_with\_retry<!-- {{#callable:curl_perform_with_retry}} -->
The `curl_perform_with_retry` function attempts to perform a CURL operation on a given URL with a specified number of retry attempts and delay between retries.
- **Inputs**:
    - `url`: A constant reference to a string representing the URL to perform the CURL operation on.
    - `curl`: A pointer to a CURL object that is used to perform the CURL operation.
    - `max_attempts`: An integer specifying the maximum number of attempts to perform the CURL operation.
    - `retry_delay_seconds`: An integer specifying the delay in seconds between retry attempts.
    - `method_name`: A constant character pointer representing the name of the method being used, for logging purposes.
- **Control Flow**:
    - Initialize `remaining_attempts` with `max_attempts`.
    - Enter a while loop that continues as long as `remaining_attempts` is greater than 0.
    - Log the attempt number and method name using `LOG_INF`.
    - Call `curl_easy_perform` with the `curl` object and store the result in `res`.
    - If `res` is `CURLE_OK`, return `true` indicating success.
    - Calculate `exponential_backoff_delay` using exponential backoff strategy based on `retry_delay_seconds` and the number of attempts made.
    - Log a warning message with `LOG_WRN` if the CURL operation failed, including the error message and retry delay.
    - Decrement `remaining_attempts`.
    - If `remaining_attempts` is 0, break out of the loop.
    - Use `std::this_thread::sleep_for` to wait for `exponential_backoff_delay` milliseconds before retrying.
    - Log an error message with `LOG_ERR` if all attempts fail.
- **Output**: Returns `true` if the CURL operation succeeds within the allowed attempts, otherwise returns `false`.


---
### common\_download\_file\_single<!-- {{#callable:common_download_file_single}} -->
The function `common_download_file_single` logs an error message and returns false, indicating that downloading a file from the internet is not possible because the code was built without CURL support.
- **Inputs**:
    - `url`: A string representing the URL of the file to be downloaded.
    - `path`: A string representing the local file path where the downloaded file should be saved.
    - `bearer_token`: A string representing the bearer token for authentication, if required.
    - `offline`: A boolean indicating whether the operation should be performed in offline mode.
- **Control Flow**:
    - The function logs an error message indicating that it cannot download a model from the internet because it was built without CURL support.
    - The function returns false, indicating the failure to download the file.
- **Output**: The function returns a boolean value, which is always false in this implementation, indicating that the file download was unsuccessful.


---
### common\_download\_file\_multiple<!-- {{#callable:common_download_file_multiple}} -->
The function `common_download_file_multiple` attempts to download multiple files from the internet but always returns false if CURL is not enabled.
- **Inputs**:
    - `urls`: A vector of pairs, where each pair contains a URL and a corresponding local file path.
    - `bearer_token`: A string representing the bearer token for authorization, if needed.
    - `offline`: A boolean indicating whether the function should operate in offline mode, which would prevent any network access.
- **Control Flow**:
    - The function logs an error message indicating that it was built without CURL and cannot download models from the internet.
    - The function returns false, indicating that the download operation cannot be performed.
- **Output**: A boolean value, which is always false in this implementation, indicating the failure to download files due to the absence of CURL support.


---
### common\_download\_model<!-- {{#callable:common_download_model}} -->
The `common_download_model` function attempts to download a model from the internet but always returns false if CURL is not enabled.
- **Inputs**:
    - `model`: A `common_params_model` object containing parameters for the model to be downloaded.
    - `bearer_token`: A `std::string` representing the bearer token for authentication, if needed.
    - `offline`: A `bool` indicating whether the download should be attempted in offline mode.
- **Control Flow**:
    - Logs an error message indicating that the function was built without CURL support.
    - Returns false, indicating the download cannot proceed.
- **Output**: Returns a boolean value `false`, indicating the model download was unsuccessful.


---
### common\_remote\_get\_content<!-- {{#callable:common_remote_get_content}} -->
The function `common_remote_get_content` attempts to retrieve content from a specified URL using CURL, returning the HTTP response code and the content as a vector of characters.
- **Inputs**:
    - `url`: A string representing the URL from which to retrieve content.
    - `params`: A `common_remote_params` structure containing additional parameters for the request, such as headers, timeout, and maximum size.
- **Control Flow**:
    - Initialize a CURL pointer and a curl_slist pointer for HTTP headers.
    - Set CURL options for the URL, progress display, and redirection following.
    - Define a write callback function to append received data to a vector of characters.
    - Set the write callback and data vector for CURL.
    - Set SSL options for Windows if applicable.
    - Set timeout and maximum file size options if specified in `params`.
    - Append a default User-Agent header and any additional headers from `params`.
    - Perform the CURL request and check for errors.
    - If the request fails, throw a runtime error with the CURL error message.
    - Retrieve the HTTP response code from the CURL info.
    - Return the response code and the content vector.
- **Output**: A `std::pair` containing a `long` for the HTTP response code and a `std::vector<char>` for the content retrieved from the URL.


---
### common\_get\_hf\_file<!-- {{#callable:common_get_hf_file}} -->
The `common_get_hf_file` function attempts to retrieve a file from a Hugging Face repository, but returns an error message and an empty result if CURL is not available.
- **Inputs**:
    - `hf_repo_with_tag`: A string representing the Hugging Face repository with an optional tag, formatted as '<user>/<model>[:quant]'.
    - `bearer_token`: A string representing the bearer token for authentication, if required.
    - `offline`: A boolean indicating whether the function should operate in offline mode, avoiding network requests.
- **Control Flow**:
    - The function logs an error message indicating that it was built without CURL and cannot download models from the internet.
    - The function returns an empty `common_hf_file_res` structure.
- **Output**: An empty `common_hf_file_res` structure, indicating no file was retrieved.


---
### common\_params\_handle\_model<!-- {{#callable:common_params_handle_model}} -->
The `common_params_handle_model` function configures and potentially downloads a model based on provided parameters, including handling default paths and URLs, and returns a result indicating if a multimodal projector was found.
- **Inputs**:
    - `model`: A reference to a `common_params_model` structure that contains model-related parameters such as repository, file, path, and URL.
    - `bearer_token`: A string representing the bearer token used for authentication when accessing remote resources.
    - `model_path_default`: A string representing the default path to use for the model if no other path is specified.
    - `offline`: A boolean indicating whether the function should operate in offline mode, avoiding network access.
- **Control Flow**:
    - Initialize a `handle_model_result` structure to store the result.
    - Check if the `hf_repo` field in the model is not empty.
    - If `hf_file` is empty, check if `path` is empty and attempt to auto-detect the file using [`common_get_hf_file`](#common_get_hf_file).
    - If auto-detection fails (empty repo or file), exit the program.
    - Update the model's `hf_repo` and `hf_file` with the auto-detected values.
    - If an `mmprojFile` is detected, update the result to indicate a found multimodal projector.
    - If `hf_file` is not empty, construct the model's URL using the endpoint, repository, and file.
    - Ensure the model's path is set, using a cache file if necessary.
    - If `url` is not empty and `path` is empty, derive the path from the URL and set it using a cache file.
    - If both `url` and `path` are empty, set the path to `model_path_default`.
    - If `url` is not empty, attempt to download the model using [`common_download_model`](#common_download_model).
    - If the download fails, log an error and exit the program.
    - Return the `handle_model_result` structure.
- **Output**: A `handle_model_result` structure indicating whether a multimodal projector was found and containing any relevant model information.
- **Functions called**:
    - [`common_get_hf_file`](#common_get_hf_file)
    - [`common_download_model`](#common_download_model)


---
### kv\_cache\_type\_from\_str<!-- {{#callable:kv_cache_type_from_str}} -->
The function `kv_cache_type_from_str` converts a string representation of a cache type to its corresponding `ggml_type` enumeration value.
- **Inputs**:
    - `s`: A string representing the name of a cache type.
- **Control Flow**:
    - Iterate over each type in the `kv_cache_types` vector.
    - For each type, check if the result of `ggml_type_name(type)` matches the input string `s`.
    - If a match is found, return the corresponding `ggml_type`.
    - If no match is found after iterating through all types, throw a `std::runtime_error` indicating an unsupported cache type.
- **Output**: Returns the `ggml_type` that corresponds to the input string `s` if found; otherwise, throws a `std::runtime_error`.
- **Functions called**:
    - [`ggml_type_name`](../ggml/src/ggml.c.driver.md#ggml_type_name)


---
### get\_all\_kv\_cache\_types<!-- {{#callable:get_all_kv_cache_types}} -->
The function `get_all_kv_cache_types` returns a comma-separated string of all key-value cache types available in the `kv_cache_types` vector.
- **Inputs**: None
- **Control Flow**:
    - Initialize an output string stream `msg`.
    - Iterate over each `type` in the `kv_cache_types` vector.
    - For each `type`, append its name (obtained via [`ggml_type_name`](../ggml/src/ggml.c.driver.md#ggml_type_name)) to `msg`.
    - If the current `type` is not the last element in `kv_cache_types`, append a comma and space to `msg`.
    - Return the constructed string from `msg`.
- **Output**: A string containing the names of all key-value cache types, separated by commas.
- **Functions called**:
    - [`ggml_type_name`](../ggml/src/ggml.c.driver.md#ggml_type_name)


---
### common\_params\_parse\_ex<!-- {{#callable:common_params_parse_ex}} -->
The `common_params_parse_ex` function parses command-line arguments and environment variables to configure a `common_params_context` object, handling various options and performing necessary post-processing and validation.
- **Inputs**:
    - `argc`: The number of command-line arguments.
    - `argv`: An array of C-style strings representing the command-line arguments.
    - `ctx_arg`: A reference to a `common_params_context` object that holds the parameters and options to be configured.
- **Control Flow**:
    - Initialize a string for argument prefix and a reference to `common_params` from `ctx_arg`.
    - Create a map to associate command-line arguments with their corresponding `common_arg` options.
    - Iterate over `ctx_arg.options` to populate the map with argument-option pairs.
    - Handle environment variables by checking if each option has a value from the environment and applying the appropriate handler function.
    - Define a lambda function `check_arg` to ensure that an argument has a subsequent value.
    - Iterate over command-line arguments starting from index 1, checking for valid arguments and applying handlers based on the argument type (void, int, string, or string-string).
    - Perform post-processing on CPU parameters and speculative parameters using `postprocess_cpu_params`.
    - Check for incompatible parameter combinations and throw exceptions if necessary.
    - Handle model and download-related configurations, including checking for multimodal projectors and downloading models if needed.
    - Process escape sequences in various string parameters if the escape flag is set.
    - Add default values to `kv_overrides` and `tensor_buft_overrides` if they are not empty.
    - Return `true` if parsing and processing are successful.
- **Output**: Returns a boolean value `true` if the parsing and processing of arguments and environment variables are successful, otherwise throws exceptions for invalid arguments or configurations.
- **Functions called**:
    - [`common_params_handle_model`](#common_params_handle_model)


---
### common\_params\_print\_usage<!-- {{#callable:common_params_print_usage}} -->
The `common_params_print_usage` function organizes and prints usage information for command-line options based on their categorization into common, sampling, and example-specific parameters.
- **Inputs**:
    - `ctx_arg`: A reference to a `common_params_context` object that contains the context and options for command-line parameters.
- **Control Flow**:
    - Define a lambda function `print_options` to print each option's string representation.
    - Initialize three vectors: `common_options`, `sparam_options`, and `specific_options` to categorize options.
    - Iterate over `ctx_arg.options` to categorize each option into `common_options`, `sparam_options`, or `specific_options` based on its properties.
    - Print the header '----- common params -----' and call `print_options` with `common_options`.
    - Print the header '----- sampling params -----' and call `print_options` with `sparam_options`.
    - Print the header '----- example-specific params -----' and call `print_options` with `specific_options`.
- **Output**: The function does not return a value; it outputs formatted usage information to the console.


---
### common\_params\_print\_completion<!-- {{#callable:common_params_print_completion}} -->
The `common_params_print_completion` function generates a bash completion script for command-line options based on the context provided.
- **Inputs**:
    - `ctx_arg`: A reference to a `common_params_context` object that contains the context for command-line options, including a list of options and the current example being used.
- **Control Flow**:
    - Initialize three vectors to categorize options: `common_options`, `sparam_options`, and `specific_options`.
    - Iterate over each option in `ctx_arg.options` and categorize them into the appropriate vector based on whether they are sampling parameters, specific to the current example, or common options.
    - Print the beginning of a bash function `_llama_completions` that sets up local variables for current and previous command-line words and initializes the `COMPREPLY` array.
    - Define a lambda function `print_options` to print the arguments of each option in a given vector.
    - Use `print_options` to print all categorized options into a single string `opts`.
    - Print a `case` statement to handle specific options like `--model`, `--grammar-file`, and `--chat-template-file`, providing file completion for specific file types.
    - For any other option, use `compgen` to generate possible completions from the `opts` string.
    - Print a list of executable names and associate the `_llama_completions` function with each using the `complete` command.
- **Output**: The function does not return a value; it outputs a bash script to standard output that can be used for command-line completion.


---
### parse\_device\_list<!-- {{#callable:parse_device_list}} -->
The `parse_device_list` function parses a comma-separated string of device names and returns a vector of GPU device pointers, throwing exceptions for invalid inputs.
- **Inputs**:
    - `value`: A string containing a comma-separated list of device names to be parsed.
- **Control Flow**:
    - Initialize an empty vector `devices` to store the parsed device pointers.
    - Split the input string `value` by commas to get individual device names in `dev_names`.
    - If `dev_names` is empty, throw an `std::invalid_argument` exception indicating no devices were specified.
    - If `dev_names` contains a single element 'none', add a `nullptr` to `devices`.
    - Otherwise, iterate over each device name in `dev_names`.
    - For each device name, retrieve the device pointer using `ggml_backend_dev_by_name`.
    - Check if the device pointer is valid and if the device type is `GGML_BACKEND_DEVICE_TYPE_GPU`.
    - If the device is invalid or not a GPU, throw an `std::invalid_argument` exception with an error message.
    - Add the valid device pointer to `devices`.
    - After processing all device names, add a `nullptr` to `devices` to signify the end of the list.
- **Output**: A vector of `ggml_backend_dev_t` pointers representing the parsed GPU devices, with a `nullptr` at the end.
- **Functions called**:
    - [`ggml_backend_dev_type`](../ggml/include/ggml-backend.h.driver.md#ggml_backend_dev_type)


---
### add\_rpc\_devices<!-- {{#callable:add_rpc_devices}} -->
The `add_rpc_devices` function registers RPC devices by splitting a comma-separated list of server endpoints and adding each as a device using a specific backend function.
- **Inputs**:
    - `servers`: A string containing a comma-separated list of RPC server endpoints.
- **Control Flow**:
    - Split the input string `servers` into a list of server endpoints using a comma as the delimiter.
    - Check if the resulting list of server endpoints is empty; if so, throw an `std::invalid_argument` exception indicating no RPC servers were specified.
    - Retrieve the RPC backend registration using `ggml_backend_reg_by_name` with the name 'RPC'.
    - Check if the RPC backend registration is valid; if not, throw an `std::invalid_argument` exception indicating failure to find the RPC backend.
    - Retrieve the function pointer for adding an RPC device using [`ggml_backend_reg_get_proc_address`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address) with the function name 'ggml_backend_rpc_add_device'.
    - Check if the function pointer is valid; if not, throw an `std::invalid_argument` exception indicating failure to find the RPC device add function.
    - Iterate over each server endpoint in the list of server endpoints.
    - For each server endpoint, call the retrieved function pointer to add the device, passing the server endpoint as a C-style string.
    - Check if the device was successfully added; if so, register the device using [`ggml_backend_device_register`](../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_device_register).
    - If the device was not successfully added, throw an `std::invalid_argument` exception indicating failure to register the RPC device.
- **Output**: The function does not return a value; it either successfully registers the devices or throws an exception if any step fails.
- **Functions called**:
    - [`ggml_backend_reg_get_proc_address`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_get_proc_address)
    - [`ggml_backend_device_register`](../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_device_register)


---
### common\_params\_parse<!-- {{#callable:common_params_parse}} -->
The `common_params_parse` function parses command-line arguments to configure common parameters for a given example, handling usage and completion requests, and managing exceptions.
- **Inputs**:
    - `argc`: The number of command-line arguments.
    - `argv`: An array of command-line argument strings.
    - `params`: A reference to a `common_params` structure that will be populated with parsed values.
    - `ex`: An enumeration value of type `llama_example` indicating the specific example context for parsing.
    - `print_usage`: A function pointer to a usage printing function, which takes `argc` and `argv` as arguments.
- **Control Flow**:
    - Initialize a `common_params_context` using [`common_params_parser_init`](#common_params_parser_init) with the provided parameters, example, and usage function.
    - Store the original parameters in `params_org` to allow reverting in case of parsing failure.
    - Attempt to parse the command-line arguments using [`common_params_parse_ex`](#common_params_parse_ex).
    - If parsing fails, revert to the original parameters and return `false`.
    - Check if the `usage` flag is set in the parsed parameters; if so, print usage information and exit.
    - Check if the `completion` flag is set in the parsed parameters; if so, print completion information and exit.
    - Catch `std::invalid_argument` exceptions, print the error message, revert parameters, and return `false`.
    - Catch other `std::exception` types, print the error message, and exit with status code 1.
    - Return `true` if parsing and processing complete successfully without exceptions.
- **Output**: Returns a boolean value: `true` if parsing and processing are successful, `false` if an error occurs during parsing.
- **Functions called**:
    - [`common_params_parser_init`](#common_params_parser_init)
    - [`common_params_parse_ex`](#common_params_parse_ex)
    - [`common_params_print_usage`](#common_params_print_usage)
    - [`common_params_print_completion`](#common_params_print_completion)


---
### list\_builtin\_chat\_templates<!-- {{#callable:list_builtin_chat_templates}} -->
The `list_builtin_chat_templates` function retrieves and returns a comma-separated string of all built-in chat templates supported by the `llama_chat_builtin_templates` function.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty vector `supported_tmpl` to store template names.
    - Call `llama_chat_builtin_templates` with `nullptr` and `0` to get the number of supported templates and resize `supported_tmpl` accordingly.
    - Call `llama_chat_builtin_templates` again with `supported_tmpl.data()` and its size to populate the vector with template names.
    - Iterate over `supported_tmpl` and append each template name to an `ostringstream` `msg`, separating them with a comma, except for the last one.
    - Return the concatenated string from `msg`.
- **Output**: A string containing the names of all built-in chat templates, separated by commas.


---
### common\_params\_parser\_init<!-- {{#callable:common_params_parser_init}} -->
The `common_params_parser_init` function initializes a `common_params_context` object with command-line options and settings based on the provided parameters, example type, and usage function.
- **Inputs**:
    - `params`: A reference to a `common_params` object that holds various configuration settings and options for the application.
    - `ex`: An enumeration value of type `llama_example` that specifies the current example or mode of operation for which the parameters are being initialized.
    - `print_usage`: A pointer to a function that takes two arguments (int, char**) and is used to print usage information for the application.
- **Control Flow**:
    - The function begins by loading all dynamic backends using `ggml_backend_load_all()`.
    - A `common_params_context` object `ctx_arg` is created and initialized with the provided `params`, `ex`, and `print_usage` function.
    - The function constructs strings `sampler_type_chars` and `sampler_type_names` by iterating over the samplers in `params.sampling.samplers`, converting each sampler to its character and string representation, respectively.
    - A lambda function `add_opt` is defined to add command-line options to `ctx_arg.options` based on the current example `ex` and common example `LLAMA_EXAMPLE_COMMON`.
    - Several [`common_arg`](arg.h.driver.md#common_argcommon_arg) objects are created and added to `ctx_arg.options` using `add_opt`, each representing a command-line option with its associated handler function and description.
- **Output**: The function returns the initialized `common_params_context` object `ctx_arg`.
- **Functions called**:
    - [`ggml_backend_load_all`](../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_load_all)
    - [`common_sampler_type_to_chr`](sampling.cpp.driver.md#common_sampler_type_to_chr)
    - [`common_sampler_type_to_str`](sampling.cpp.driver.md#common_sampler_type_to_str)
    - [`common_arg::common_arg`](arg.h.driver.md#common_argcommon_arg)
    - [`read_file`](#read_file)
    - [`common_sampler_types_from_names`](sampling.cpp.driver.md#common_sampler_types_from_names)
    - [`common_sampler_types_from_chars`](sampling.cpp.driver.md#common_sampler_types_from_chars)
    - [`json_schema_to_grammar`](json-schema-to-grammar.cpp.driver.md#json_schema_to_grammar)
    - [`get_all_kv_cache_types`](#get_all_kv_cache_types)
    - [`ggml_type_name`](../ggml/src/ggml.c.driver.md#ggml_type_name)
    - [`kv_cache_type_from_str`](#kv_cache_type_from_str)
    - [`add_rpc_devices`](#add_rpc_devices)
    - [`parse_device_list`](#parse_device_list)
    - [`ggml_backend_dev_count`](../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_dev_count)
    - [`ggml_backend_dev_type`](../ggml/include/ggml-backend.h.driver.md#ggml_backend_dev_type)
    - [`ggml_backend_reg_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_reg_name)
    - [`ggml_backend_dev_memory`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_memory)
    - [`ggml_backend_dev_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_name)
    - [`ggml_backend_dev_description`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_dev_description)
    - [`ggml_backend_buft_name`](../ggml/src/ggml-backend.cpp.driver.md#ggml_backend_buft_name)
    - [`list_builtin_chat_templates`](#list_builtin_chat_templates)
    - [`common_log_pause`](log.cpp.driver.md#common_log_pause)
    - [`common_log_main`](log.cpp.driver.md#common_log_main)
    - [`common_log_set_file`](log.cpp.driver.md#common_log_set_file)
    - [`common_log_set_colors`](log.cpp.driver.md#common_log_set_colors)
    - [`common_log_set_verbosity_thold`](log.cpp.driver.md#common_log_set_verbosity_thold)
    - [`common_log_set_prefix`](log.cpp.driver.md#common_log_set_prefix)
    - [`common_log_set_timestamps`](log.cpp.driver.md#common_log_set_timestamps)


