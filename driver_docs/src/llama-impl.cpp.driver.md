# Purpose
This C++ source code file provides a set of utility functions and structures primarily focused on logging, string manipulation, and data formatting. The file includes several components that work together to facilitate logging operations, such as setting custom log callbacks and formatting log messages. The `llama_logger_state` structure maintains the state of the logging system, including the current log callback and associated user data. The [`llama_log_set`](#llama_log_set) function allows users to set a custom logging callback, while [`llama_log_internal`](#llama_log_internal) and [`llama_log_internal_v`](#llama_log_internal_v) handle the formatting and dispatching of log messages. The default logging behavior is defined by [`llama_log_callback_default`](#llama_log_callback_default), which outputs messages to standard error.

In addition to logging, the file includes utility functions for string manipulation and data formatting. The [`replace_all`](#replace_all) function replaces occurrences of a substring within a string, while the [`format`](#format) function provides a way to format strings using a printf-style interface. The file also includes functions for formatting tensor shapes and converting data types to strings, such as [`llama_format_tensor_shape`](#llama_format_tensor_shape) and [`gguf_data_to_str`](#gguf_data_to_str). These functions are particularly useful for handling and displaying data in a structured format, such as when working with tensors or key-value pairs in a GGUF context. Overall, the file serves as a utility library that can be integrated into larger projects to provide enhanced logging and data handling capabilities.
# Imports and Dependencies

---
- `llama-impl.h`
- `gguf.h`
- `llama.h`
- `cinttypes`
- `climits`
- `cstdarg`
- `cstring`
- `vector`
- `sstream`


# Global Variables

---
### g\_logger\_state
- **Type**: `llama_logger_state`
- **Description**: The `g_logger_state` is a static global variable of type `llama_logger_state`, which is a structure that holds logging-related information. This structure contains a logging callback function and a pointer to user data associated with the logging process.
- **Use**: This variable is used to manage and store the current state of the logging system, including the callback function and user data, which are utilized during logging operations.


# Data Structures

---
### llama\_logger\_state<!-- {{#data_structure:llama_logger_state}} -->
- **Type**: `struct`
- **Members**:
    - `log_callback`: A function pointer for logging, initialized to a default callback.
    - `log_callback_user_data`: A pointer to user data associated with the log callback, initialized to nullptr.
- **Description**: The `llama_logger_state` struct is designed to manage the state of logging within the application. It holds a callback function pointer, `log_callback`, which is used to handle log messages, and a `log_callback_user_data` pointer, which can store additional user-defined data to be passed to the callback function. This structure allows for flexible logging configurations by enabling the user to specify custom logging behavior and associated data.


---
### time\_meas<!-- {{#data_structure:time_meas}} -->
- **Description**: [See definition](llama-impl.h.driver.md#time_meas)
- **Member Functions**:
    - [`time_meas::time_meas`](#time_meastime_meas)
    - [`time_meas::~time_meas`](#time_meastime_meas)

**Methods**

---
#### time\_meas::time\_meas<!-- {{#callable:time_meas::time_meas}} -->
The `time_meas` constructor initializes a timing measurement object, optionally disabling timing, and associates it with an accumulator for elapsed time.
- **Inputs**:
    - `t_acc`: A reference to an `int64_t` variable that accumulates the elapsed time in microseconds.
    - `disable`: A boolean flag indicating whether to disable timing; if true, timing is disabled and `t_start_us` is set to -1.
- **Control Flow**:
    - The constructor checks if the `disable` flag is true.
    - If `disable` is true, `t_start_us` is set to -1, indicating that timing is disabled.
    - If `disable` is false, `t_start_us` is initialized with the current time in microseconds using `ggml_time_us()`.
    - The reference to the elapsed time accumulator `t_acc` is stored in the object.
- **Output**: The constructor does not return a value; it initializes the `time_meas` object.
- **See also**: [`time_meas`](llama-impl.h.driver.md#time_meas)  (Data Structure)


---
#### time\_meas::\~time\_meas<!-- {{#callable:time_meas::~time_meas}} -->
The destructor `~time_meas` finalizes the timing measurement by updating the accumulated time if the measurement was started.
- **Inputs**: None
- **Control Flow**:
    - Check if `t_start_us` is non-negative, indicating that timing was started.
    - If true, calculate the elapsed time since `t_start_us` using `ggml_time_us()` and add it to `t_acc`.
- **Output**: The function does not return any value as it is a destructor.
- **See also**: [`time_meas`](llama-impl.h.driver.md#time_meas)  (Data Structure)



# Functions

---
### llama\_log\_set<!-- {{#callable:llama_log_set}} -->
The `llama_log_set` function configures the logging callback and user data for the logging system.
- **Inputs**:
    - `log_callback`: A function pointer of type `ggml_log_callback` that specifies the logging callback function to be used.
    - `user_data`: A pointer to user-defined data that will be passed to the logging callback function.
- **Control Flow**:
    - Call the [`ggml_log_set`](../ggml/src/ggml.c.driver.md#ggml_log_set) function with `log_callback` and `user_data` to set the global logging configuration.
    - Assign `log_callback` to `g_logger_state.log_callback` if it is not null; otherwise, assign `llama_log_callback_default`.
    - Assign `user_data` to `g_logger_state.log_callback_user_data`.
- **Output**: This function does not return any value.
- **Functions called**:
    - [`ggml_log_set`](../ggml/src/ggml.c.driver.md#ggml_log_set)


---
### llama\_log\_internal\_v<!-- {{#callable:llama_log_internal_v}} -->
The `llama_log_internal_v` function formats a log message using a variable argument list and sends it to a logging callback, handling both small and large messages.
- **Inputs**:
    - `level`: The logging level, indicating the severity or type of the log message.
    - `format`: A C-style string that contains the format string for the log message.
    - `args`: A `va_list` of arguments that correspond to the format specifiers in the format string.
- **Control Flow**:
    - A copy of the `va_list` `args` is made to `args_copy` to allow for multiple traversals of the argument list.
    - A buffer of 128 characters is allocated to hold the formatted message.
    - The `vsnprintf` function is used to format the message into the buffer using the `args` list.
    - If the formatted message fits within 128 characters, it is sent to the logging callback using `g_logger_state.log_callback`.
    - If the message exceeds 128 characters, a new buffer is dynamically allocated to fit the entire message.
    - The message is formatted again into the new buffer using `args_copy`, and then sent to the logging callback.
    - The dynamically allocated buffer is deleted after use to free memory.
    - The `va_end` function is called on `args_copy` to clean up the copied argument list.
- **Output**: The function does not return a value; it outputs the formatted log message to the specified logging callback.


---
### llama\_log\_internal<!-- {{#callable:llama_log_internal}} -->
The `llama_log_internal` function logs a formatted message at a specified log level using a variable argument list.
- **Inputs**:
    - `level`: The log level of type `ggml_log_level` indicating the severity or type of the log message.
    - `format`: A C-style string that contains the format of the log message, similar to printf-style formatting.
    - `...`: A variable number of arguments that are used in conjunction with the format string to create the log message.
- **Control Flow**:
    - Initialize a `va_list` variable `args` to handle the variable arguments.
    - Start processing the variable arguments using `va_start`, with `format` as the last fixed argument.
    - Call the helper function [`llama_log_internal_v`](#llama_log_internal_v) with the log level, format string, and the initialized `va_list` to perform the actual logging.
    - End the processing of the variable arguments using `va_end`.
- **Output**: This function does not return a value; it performs logging as a side effect.
- **Functions called**:
    - [`llama_log_internal_v`](#llama_log_internal_v)


---
### llama\_log\_callback\_default<!-- {{#callable:llama_log_callback_default}} -->
The `llama_log_callback_default` function writes a given log message to the standard error stream.
- **Inputs**:
    - `level`: The log level of the message, which is not used in this function.
    - `text`: The log message to be written to the standard error stream.
    - `user_data`: User-defined data, which is not used in this function.
- **Control Flow**:
    - The function begins by explicitly ignoring the `level` and `user_data` parameters using `(void)` casts.
    - It then writes the `text` message to the standard error stream using `fputs`.
    - Finally, it flushes the standard error stream to ensure the message is immediately outputted.
- **Output**: The function does not return any value.


---
### replace\_all<!-- {{#callable:replace_all}} -->
The `replace_all` function replaces all occurrences of a specified substring within a given string with another substring.
- **Inputs**:
    - `s`: A reference to the string in which occurrences of the search substring will be replaced.
    - `search`: The substring to search for within the string `s`.
    - `replace`: The substring to replace each occurrence of the search substring with.
- **Control Flow**:
    - Check if the `search` string is empty; if so, return immediately as no replacements are needed.
    - Initialize a `builder` string with reserved space equal to the length of `s` to efficiently build the new string.
    - Initialize `pos` and `last_pos` to track positions within the string `s`.
    - Enter a loop to find each occurrence of `search` in `s` starting from `last_pos`.
    - For each occurrence, append the substring from `last_pos` to `pos` from `s` to `builder`, followed by the `replace` string.
    - Update `last_pos` to the position immediately after the found `search` substring.
    - Continue the loop until no more occurrences are found.
    - Append the remaining part of `s` from `last_pos` to the end to `builder`.
    - Move the contents of `builder` back to `s`, effectively replacing all occurrences of `search` with `replace`.
- **Output**: The function modifies the input string `s` in place, replacing all occurrences of `search` with `replace`.


---
### format<!-- {{#callable:format}} -->
The `format` function creates a formatted string using a format specifier and a variable number of arguments.
- **Inputs**:
    - `fmt`: A C-style string that contains the format specifier.
    - `...`: A variable number of arguments that are used to replace format specifiers in the format string.
- **Control Flow**:
    - Initialize a `va_list` variable `ap` to access the variable arguments.
    - Copy `ap` to another `va_list` variable `ap2` to preserve the original list for reuse.
    - Use `vsnprintf` with a `NULL` buffer to calculate the size needed for the formatted string.
    - Assert that the calculated size is non-negative and less than `INT_MAX`.
    - Create a buffer of the calculated size plus one for the null terminator.
    - Use `vsnprintf` again to write the formatted string into the buffer using `ap2`.
    - Assert that the size of the written string matches the calculated size.
    - End the use of both `va_list` variables `ap` and `ap2`.
- **Output**: A `std::string` containing the formatted output.


---
### llama\_format\_tensor\_shape<!-- {{#callable:llama_format_tensor_shape}} -->
The function `llama_format_tensor_shape` formats the shape of a tensor into a string representation.
- **Inputs**:
    - `t`: A pointer to a `ggml_tensor` structure, which contains the dimensions of the tensor to be formatted.
- **Control Flow**:
    - Initialize a character buffer `buf` with a size of 256 to store the formatted string.
    - Use `snprintf` to format the first dimension of the tensor `t->ne[0]` into the buffer `buf`.
    - Iterate over the remaining dimensions of the tensor from index 1 to `GGML_MAX_DIMS - 1`.
    - For each dimension, append its formatted value to the buffer `buf` using `snprintf`, ensuring not to exceed the buffer size.
    - Return the formatted string stored in `buf`.
- **Output**: A `std::string` containing the formatted shape of the tensor, with each dimension separated by a comma.


---
### gguf\_data\_to\_str<!-- {{#callable:gguf_data_to_str}} -->
The function `gguf_data_to_str` converts a data element of a specified type from a given array to its string representation.
- **Inputs**:
    - `type`: An enumeration value of type `gguf_type` indicating the data type of the element to be converted.
    - `data`: A pointer to the data array containing elements of the specified type.
    - `i`: An integer index specifying the position of the element in the data array to be converted to a string.
- **Control Flow**:
    - The function uses a switch statement to determine the type of the data element based on the `type` parameter.
    - For each case corresponding to a specific `gguf_type`, the function casts the `data` pointer to the appropriate type and accesses the element at index `i`.
    - The accessed element is then converted to a string using `std::to_string` for numeric types or a ternary operator for boolean types.
    - If the `type` does not match any known `gguf_type`, the function returns a formatted string indicating an unknown type.
- **Output**: A `std::string` representing the value of the data element at index `i` in the array, or an error message if the type is unknown.
- **Functions called**:
    - [`format`](#format)


---
### gguf\_kv\_to\_str<!-- {{#callable:gguf_kv_to_str}} -->
The function `gguf_kv_to_str` converts a key-value pair from a GGUF context into a string representation based on its type.
- **Inputs**:
    - `ctx_gguf`: A pointer to a `gguf_context` structure, which contains the key-value pairs to be processed.
    - `i`: An integer index specifying which key-value pair in the context to convert to a string.
- **Control Flow**:
    - Retrieve the type of the key-value pair at index `i` using [`gguf_get_kv_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_kv_type).
    - Use a switch statement to handle different types of key-value pairs.
    - If the type is `GGUF_TYPE_STRING`, return the string value using [`gguf_get_val_str`](../ggml/src/gguf.cpp.driver.md#gguf_get_val_str).
    - If the type is `GGUF_TYPE_ARRAY`, determine the array's element type and number of elements.
    - For arrays of strings, iterate over each element, escape special characters, and append to a string stream.
    - For arrays of other types, convert each element to a string using [`gguf_data_to_str`](#gguf_data_to_str) and append to the string stream.
    - If the array type is `GGUF_TYPE_ARRAY`, append '???' to the string stream for each element.
    - For other types, convert the value to a string using [`gguf_data_to_str`](#gguf_data_to_str).
- **Output**: A string representation of the key-value pair at the specified index, formatted according to its type.
- **Functions called**:
    - [`gguf_get_kv_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_kv_type)
    - [`gguf_get_val_str`](../ggml/src/gguf.cpp.driver.md#gguf_get_val_str)
    - [`gguf_get_arr_type`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_type)
    - [`gguf_get_arr_n`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_n)
    - [`gguf_get_arr_data`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_data)
    - [`gguf_get_arr_str`](../ggml/src/gguf.cpp.driver.md#gguf_get_arr_str)
    - [`replace_all`](#replace_all)
    - [`gguf_data_to_str`](#gguf_data_to_str)
    - [`gguf_get_val_data`](../ggml/src/gguf.cpp.driver.md#gguf_get_val_data)


