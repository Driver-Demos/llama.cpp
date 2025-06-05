# Purpose
This C++ source code file is designed to facilitate command-line interface (CLI) argument parsing and remote content retrieval. It provides a structured approach to handling CLI arguments through the [`common_arg`](#common_argcommon_arg) and [`common_params_context`](#common_params_contextcommon_params_context) structures. The [`common_arg`](#common_argcommon_arg) structure defines various properties and handlers for CLI arguments, allowing for flexible argument parsing, including support for multiple argument values and environment variable integration. The [`common_params_context`](#common_params_contextcommon_params_context) structure manages the context for parsing, including the specific example being executed and the parameters involved. The file also includes functions like `common_params_parse` and `common_params_parser_init` to initialize and parse CLI arguments, ensuring that invalid arguments trigger specific usage messages.

Additionally, the file includes functionality for remote content retrieval through the `common_remote_params` structure and the `common_remote_get_content` function. This part of the code is responsible for fetching content from a specified URL, with configurable parameters such as headers, timeout, and maximum response size. The function returns a pair consisting of the HTTP status code and the raw response body, enabling further processing of remote data. Overall, this file provides a cohesive set of utilities for both CLI argument management and remote data access, making it a versatile component in applications requiring these capabilities.
# Imports and Dependencies

---
- `common.h`
- `set`
- `string`
- `vector`


# Data Structures

---
### common\_arg<!-- {{#data_structure:common_arg}} -->
- **Type**: `struct`
- **Members**:
    - `examples`: A set of llama_example enums representing examples to include.
    - `excludes`: A set of llama_example enums representing examples to exclude.
    - `args`: A vector of C-style strings representing command-line arguments.
    - `value_hint`: A C-style string providing help text or an example for the argument value.
    - `value_hint_2`: A C-style string for a second argument value hint.
    - `env`: A C-style string representing the environment variable associated with the argument.
    - `help`: A string containing help text for the argument.
    - `is_sparam`: A boolean indicating if the current argument is a sampling parameter.
    - `handler_void`: A function pointer for handling arguments with no additional parameters.
    - `handler_string`: A function pointer for handling arguments with a single string parameter.
    - `handler_str_str`: A function pointer for handling arguments with two string parameters.
    - `handler_int`: A function pointer for handling arguments with an integer parameter.
- **Description**: The `common_arg` struct is designed for command-line interface (CLI) argument parsing, encapsulating various properties and handlers for arguments. It includes sets for examples and exclusions, vectors for argument strings, and pointers for handling different types of argument values. The struct supports multiple constructors to initialize arguments with different types of handlers, and provides methods to set examples, exclusions, environment variables, and sampling parameters. It also includes functionality to check if an argument is in examples or exclusions, retrieve values from environment variables, and convert the argument details to a string representation.
- **Member Functions**:
    - [`common_arg::set_examples`](arg.cpp.driver.md#common_argset_examples)
    - [`common_arg::set_excludes`](arg.cpp.driver.md#common_argset_excludes)
    - [`common_arg::set_env`](arg.cpp.driver.md#common_argset_env)
    - [`common_arg::set_sparam`](arg.cpp.driver.md#common_argset_sparam)
    - [`common_arg::in_example`](arg.cpp.driver.md#common_argin_example)
    - [`common_arg::is_exclude`](arg.cpp.driver.md#common_argis_exclude)
    - [`common_arg::get_value_from_env`](arg.cpp.driver.md#common_argget_value_from_env)
    - [`common_arg::has_value_from_env`](arg.cpp.driver.md#common_arghas_value_from_env)
    - [`common_arg::to_string`](arg.cpp.driver.md#common_argto_string)
    - [`common_arg::common_arg`](#common_argcommon_arg)
    - [`common_arg::common_arg`](#common_argcommon_arg)
    - [`common_arg::common_arg`](#common_argcommon_arg)
    - [`common_arg::common_arg`](#common_argcommon_arg)

**Methods**

---
#### common\_arg::common\_arg<!-- {{#callable:common_arg::common_arg}} -->
The `common_arg` constructor initializes a `common_arg` object with a list of argument names, a value hint, help text, and a handler function that processes a string argument.
- **Inputs**:
    - `args`: A list of C-style string argument names that the `common_arg` object will recognize.
    - `value_hint`: A C-style string providing a hint or example for the argument's value.
    - `help`: A string containing help text that describes the purpose or usage of the argument.
    - `handler`: A pointer to a function that takes a `common_params` reference and a `std::string`, used to handle the argument's value.
- **Control Flow**:
    - The constructor initializes the `args` member with the provided list of argument names.
    - It sets the `value_hint` member to the provided value hint string.
    - The `help` member is initialized with the provided help text.
    - The `handler_string` member is set to the provided handler function pointer, which processes a string argument.
- **Output**: An initialized `common_arg` object with specified argument names, value hint, help text, and a handler function for string arguments.
- **See also**: [`common_arg`](#common_arg)  (Data Structure)


---
#### common\_arg::common\_arg<!-- {{#callable:common_arg::common_arg}} -->
The `common_arg` constructor initializes a `common_arg` object with a list of argument names, a value hint, help text, and a handler function that processes an integer value.
- **Inputs**:
    - `args`: A list of argument names represented as an initializer list of constant character pointers.
    - `value_hint`: A constant character pointer providing a hint or example for the argument value.
    - `help`: A string containing help text for the argument.
    - `handler`: A function pointer to a handler that takes a `common_params` reference and an integer as parameters.
- **Control Flow**:
    - The constructor initializes the `args` member with the provided list of argument names.
    - The `value_hint` member is set to the provided value hint.
    - The `help` member is initialized with the provided help text.
    - The `handler_int` member is assigned the provided handler function pointer.
- **Output**: An instance of the `common_arg` class is created and initialized with the specified arguments and handler.
- **See also**: [`common_arg`](#common_arg)  (Data Structure)


---
#### common\_arg::common\_arg<!-- {{#callable:common_arg::common_arg}} -->
The `common_arg` constructor initializes a `common_arg` object with a list of argument names, a help description, and a handler function that takes a `common_params` reference.
- **Inputs**:
    - `args`: An initializer list of constant character pointers representing the argument names.
    - `help`: A string providing a help description for the argument.
    - `handler`: A pointer to a function that takes a `common_params` reference and performs some operation.
- **Control Flow**:
    - The constructor initializes the `args` member with the provided list of argument names.
    - The `help` member is set to the provided help string.
    - The `handler_void` member is assigned the provided handler function pointer.
- **Output**: An instance of the `common_arg` struct is initialized with the specified arguments, help text, and handler function.
- **See also**: [`common_arg`](#common_arg)  (Data Structure)


---
#### common\_arg::common\_arg<!-- {{#callable:common_arg::common_arg}} -->
The `common_arg` constructor initializes an instance of the `common_arg` structure with two value hints and a handler function that takes two string arguments.
- **Inputs**:
    - `args`: An initializer list of constant character pointers representing argument names.
    - `value_hint`: A constant character pointer providing help text or an example for the first argument value.
    - `value_hint_2`: A constant character pointer providing help text or an example for the second argument value.
    - `help`: A string containing help text for the argument.
    - `handler`: A function pointer to a handler that takes a `common_params` reference and two string arguments.
- **Control Flow**:
    - The constructor initializes the `args` member with the provided initializer list of argument names.
    - It sets the `value_hint` member to the provided first value hint.
    - It sets the `value_hint_2` member to the provided second value hint.
    - The `help` member is initialized with the provided help string.
    - The `handler_str_str` member is set to the provided handler function pointer.
- **Output**: An instance of the `common_arg` structure is initialized with the provided arguments and handler.
- **See also**: [`common_arg`](#common_arg)  (Data Structure)



---
### common\_params\_context<!-- {{#data_structure:common_params_context}} -->
- **Type**: `struct`
- **Members**:
    - `ex`: An enum of type llama_example initialized to LLAMA_EXAMPLE_COMMON.
    - `params`: A reference to a common_params object.
    - `options`: A vector containing common_arg objects.
    - `print_usage`: A function pointer for printing usage information, initialized to nullptr.
- **Description**: The `common_params_context` struct is designed to encapsulate the context for common parameters used in command-line interface (CLI) argument parsing. It holds an enumeration value `ex` to specify the example type, a reference to a `common_params` object for parameter management, a vector of `common_arg` objects to store various command-line options, and a function pointer `print_usage` for displaying usage information. This struct is primarily used to facilitate the initialization and parsing of CLI arguments, ensuring that the parameters are correctly managed and that appropriate usage information is displayed when needed.
- **Member Functions**:
    - [`common_params_context::common_params_context`](#common_params_contextcommon_params_context)

**Methods**

---
#### common\_params\_context::common\_params\_context<!-- {{#callable:common_params_context::common_params_context}} -->
The `common_params_context` constructor initializes a `common_params_context` object with a reference to a `common_params` object.
- **Inputs**:
    - `params`: A reference to a `common_params` object that will be used to initialize the `params` member of the `common_params_context` object.
- **Control Flow**:
    - The constructor takes a reference to a `common_params` object as an argument.
    - It initializes the `params` member of the `common_params_context` object with the provided `common_params` reference.
- **Output**: An instance of `common_params_context` with its `params` member initialized to the provided `common_params` reference.
- **See also**: [`common_params_context`](#common_params_context)  (Data Structure)



---
### common\_remote\_params<!-- {{#data_structure:common_remote_params}} -->
- **Type**: `struct`
- **Members**:
    - `headers`: A vector of strings representing HTTP headers to be included in the remote request.
    - `timeout`: A long integer specifying the timeout duration in seconds for the remote request; 0 indicates no timeout.
    - `max_size`: A long integer defining the maximum size of the response in bytes; 0 means unlimited, with a maximum of 2GB.
- **Description**: The `common_remote_params` struct is designed to encapsulate parameters for making remote HTTP requests, including headers, timeout, and maximum response size. It provides a flexible way to configure these aspects of a request, allowing for customization of HTTP headers, setting a timeout to prevent hanging requests, and limiting the size of the response to manage resource usage effectively.


