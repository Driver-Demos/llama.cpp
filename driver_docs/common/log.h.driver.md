# Purpose
This C header file provides a comprehensive logging utility designed to facilitate structured and configurable logging within an application. The file defines a set of macros and functions that allow developers to log messages at various levels of severity, such as info, warning, error, and debug. It supports features like verbosity control, color-coded output, and optional prefixes and timestamps for log entries. The logging system is built around a `common_log` structure, which manages log messages through an internal worker thread, allowing for asynchronous logging operations. The file also includes macros to conditionally compile log messages based on the current verbosity threshold, optimizing performance by avoiding unnecessary computation of log arguments when they are not needed.

The header file is intended to be included in other C source files, providing a public API for logging operations. It defines several macros for different log levels, which internally use the [`common_log_add`](#common_log_add) function to format and output log messages. The file also includes platform-specific attributes to ensure compatibility with different compilers, such as GCC and MinGW. The logging system is not thread-safe, as indicated by the comments, which means that care must be taken when using it in multi-threaded applications. Overall, this file offers a robust and flexible logging solution that can be easily integrated into C projects to enhance debugging and monitoring capabilities.
# Imports and Dependencies

---
- `ggml.h`


# Global Variables

---
### common\_log\_verbosity\_thold
- **Type**: `int`
- **Description**: The `common_log_verbosity_thold` is an external integer variable that represents the verbosity threshold for logging operations. It is used to determine whether a log message should be processed based on its verbosity level.
- **Use**: This variable is used in conjunction with logging macros to filter out log messages that have a verbosity level higher than the threshold set by `common_log_verbosity_thold`.


---
### common\_log\_init
- **Type**: `function pointer`
- **Description**: The `common_log_init` is a function that returns a pointer to a `struct common_log`. This function is likely responsible for initializing a logging system, setting up necessary resources or configurations for logging operations.
- **Use**: This function is used to initialize and obtain a new instance of the `common_log` structure for logging purposes.


---
### common\_log\_main
- **Type**: `function pointer`
- **Description**: The `common_log_main` function returns a pointer to a `common_log` structure, which acts as a singleton instance for logging purposes. This function ensures that there is a single, globally accessible logging instance that automatically cleans up upon program exit.
- **Use**: It is used to obtain the main logging instance for the application, facilitating centralized logging operations.


# Function Declarations (Public API)

---
### common\_log\_set\_verbosity\_thold<!-- {{#callable_declaration:common_log_set_verbosity_thold}} -->
Set the verbosity threshold for logging.
- **Description**: Use this function to define the minimum verbosity level required for log messages to be processed. This function is useful when you want to control the amount of log output based on the verbosity level. It is important to note that this function is not thread-safe, so it should be used in a context where concurrent access is managed or not a concern.
- **Inputs**:
    - `verbosity`: An integer representing the verbosity threshold. Log messages with a verbosity level equal to or lower than this value will be processed. There are no explicit constraints on the range of this parameter, but it should align with the verbosity levels used in the logging system.
- **Output**: None
- **See also**: [`common_log_set_verbosity_thold`](log.cpp.driver.md#common_log_set_verbosity_thold)  (Implementation)


---
### common\_log\_init<!-- {{#callable_declaration:common_log_init}} -->
Initialize a new logging instance.
- **Description**: Use this function to create a new instance of a logging structure, which can be used to manage and output log messages. This function is typically called at the beginning of a logging session to set up the necessary resources for logging operations. The returned logging instance can be configured and used with other logging functions to control log output, verbosity, and formatting. Ensure to manage the lifecycle of the logging instance appropriately, including freeing resources when no longer needed.
- **Inputs**: None
- **Output**: Returns a pointer to a newly initialized `common_log` structure, or `NULL` if initialization fails.
- **See also**: [`common_log_init`](log.cpp.driver.md#common_log_init)  (Implementation)


---
### common\_log\_main<!-- {{#callable_declaration:common_log_main}} -->
Retrieve a singleton instance of the common log.
- **Description**: This function provides access to a singleton instance of the common log, which is used for logging messages throughout the application. It ensures that only one instance of the log is created and returned, making it suitable for use in scenarios where a centralized logging mechanism is required. The log instance is automatically managed and destroyed upon application exit, so there is no need for the caller to manually free it. This function should be used whenever access to the common log is needed.
- **Inputs**: None
- **Output**: Returns a pointer to a singleton instance of the common_log structure.
- **See also**: [`common_log_main`](log.cpp.driver.md#common_log_main)  (Implementation)


---
### common\_log\_pause<!-- {{#callable_declaration:common_log_pause}} -->
Pause the logging worker thread.
- **Description**: Use this function to temporarily pause the logging worker thread associated with the given `common_log` instance. This is useful when you want to stop processing and discarding incoming log messages. It is important to note that this function is not thread-safe, so it should be called in a context where thread safety is ensured, such as when no other threads are interacting with the logging system.
- **Inputs**:
    - `log`: A pointer to a `common_log` structure representing the logging instance to be paused. This parameter must not be null, and the caller retains ownership of the `common_log` instance. Passing a null pointer will result in undefined behavior.
- **Output**: None
- **See also**: [`common_log_pause`](log.cpp.driver.md#common_log_pause)  (Implementation)


---
### common\_log\_resume<!-- {{#callable_declaration:common_log_resume}} -->
Resume the worker thread of a logging system.
- **Description**: Use this function to resume the worker thread of a logging system that has been previously paused. This is necessary to allow the logging system to continue processing and outputting log messages. It is important to note that this function is not thread-safe, so it should be called in a context where thread safety is ensured. This function should be used after a call to `common_log_pause` when you want to restart the logging operations.
- **Inputs**:
    - `log`: A pointer to a `struct common_log` representing the logging system. This parameter must not be null, and the caller retains ownership. Passing a null pointer will result in undefined behavior.
- **Output**: None
- **See also**: [`common_log_resume`](log.cpp.driver.md#common_log_resume)  (Implementation)


---
### common\_log\_free<!-- {{#callable_declaration:common_log_free}} -->
Frees the resources associated with a common_log instance.
- **Description**: Use this function to release the resources allocated for a common_log instance when it is no longer needed. This function should be called to prevent memory leaks after the logging operations are complete. Ensure that the log parameter is a valid pointer to a common_log structure that was previously initialized. Passing a null pointer to this function is safe and will have no effect.
- **Inputs**:
    - `log`: A pointer to a common_log structure that needs to be freed. Must be a valid pointer obtained from common_log_init or common_log_main. Passing a null pointer is allowed and will result in no operation.
- **Output**: None
- **See also**: [`common_log_free`](log.cpp.driver.md#common_log_free)  (Implementation)


---
### common\_log\_add<!-- {{#callable_declaration:common_log_add}} -->
Logs a formatted message with a specified log level.
- **Description**: Use this function to log messages with varying levels of importance, such as info, warning, error, or debug, to a logging system. The function requires a valid logging context and a format string, followed by any additional arguments needed for the format. It is essential to ensure that the logging system is properly initialized before calling this function. The function is not thread-safe, so care should be taken when using it in multi-threaded environments.
- **Inputs**:
    - `log`: A pointer to a 'struct common_log' that represents the logging context. Must not be null, and the logging system should be initialized before use.
    - `level`: An 'enum ggml_log_level' value indicating the severity or importance of the log message. Valid values are defined in the 'ggml_log_level' enumeration.
    - `fmt`: A C-style format string that specifies how subsequent arguments are converted for output. Must not be null.
    - `...`: Additional arguments required by the format string. These must match the types expected by the format specifiers in 'fmt'.
- **Output**: None
- **See also**: [`common_log_add`](log.cpp.driver.md#common_log_add)  (Implementation)


---
### common\_log\_set\_file<!-- {{#callable_declaration:common_log_set_file}} -->
Sets the log file for the specified logging instance.
- **Description**: Use this function to specify the file where log messages should be written for a given logging instance. This function is not thread-safe and should be called when the logging instance is not being accessed by other threads. It is useful for directing log output to a specific file, which can be helpful for persistent logging or debugging purposes. Ensure that the logging instance is properly initialized before calling this function.
- **Inputs**:
    - `log`: A pointer to a 'struct common_log' instance. This must be a valid, initialized logging instance and must not be null. The caller retains ownership.
    - `file`: A pointer to a null-terminated string representing the file path where logs should be written. This can be null to disable file logging. The caller retains ownership of the string.
- **Output**: None
- **See also**: [`common_log_set_file`](log.cpp.driver.md#common_log_set_file)  (Implementation)


---
### common\_log\_set\_colors<!-- {{#callable_declaration:common_log_set_colors}} -->
Enable or disable colored output in the log.
- **Description**: Use this function to control whether log messages are displayed with color codes. This can be useful for improving readability in terminal outputs that support ANSI color codes. The function must be called with a valid `common_log` instance. It is not thread-safe, so ensure that it is not called concurrently with other operations on the same log instance.
- **Inputs**:
    - `log`: A pointer to a `common_log` structure. This must be a valid, initialized log instance. The caller retains ownership and must ensure it is not null.
    - `colors`: A boolean value indicating whether to enable (true) or disable (false) colored output in the log.
- **Output**: None
- **See also**: [`common_log_set_colors`](log.cpp.driver.md#common_log_set_colors)  (Implementation)


---
### common\_log\_set\_prefix<!-- {{#callable_declaration:common_log_set_prefix}} -->
Set the prefix option for log messages.
- **Description**: Use this function to enable or disable the prefix in log messages for a given logging instance. This function is useful when you want to control whether log messages should include a prefix, which can be helpful for distinguishing log entries or adding context. It must be called with a valid logging instance and is not thread-safe, so ensure that concurrent access is managed appropriately.
- **Inputs**:
    - `log`: A pointer to a `struct common_log` instance. Must not be null. The caller retains ownership and is responsible for ensuring the instance is valid and not being accessed concurrently by other threads.
    - `prefix`: A boolean value indicating whether to enable (true) or disable (false) the prefix for log messages.
- **Output**: None
- **See also**: [`common_log_set_prefix`](log.cpp.driver.md#common_log_set_prefix)  (Implementation)


---
### common\_log\_set\_timestamps<!-- {{#callable_declaration:common_log_set_timestamps}} -->
Enable or disable timestamp output in log messages.
- **Description**: Use this function to control whether timestamps are included in the prefix of log messages. This can be useful for tracking the timing of events in the log output. The function must be called with a valid `common_log` instance. It is not thread-safe, so ensure that it is not called concurrently with other operations on the same log instance.
- **Inputs**:
    - `log`: A pointer to a `common_log` structure. Must not be null. The caller retains ownership and is responsible for ensuring the log is valid and not used concurrently.
    - `timestamps`: A boolean value indicating whether timestamps should be included (true) or not (false) in the log message prefix.
- **Output**: None
- **See also**: [`common_log_set_timestamps`](log.cpp.driver.md#common_log_set_timestamps)  (Implementation)


