# Purpose
This C++ source code file implements a logging system designed to handle and manage log messages with varying levels of severity, such as debug, info, warning, and error. The code provides a structured way to log messages with optional features like colored output, timestamping, and message prefixing. The core functionality is encapsulated within the [`common_log`](#common_logcommon_log) class, which manages a ring buffer of log entries and a worker thread that processes these entries. The log entries are stored in a vector, and the system can dynamically expand the buffer size if needed. The logging system supports both console and file outputs, allowing for flexible logging configurations.

The file defines a public API for interacting with the logging system, including functions to initialize, pause, resume, and free the logging resources. It also provides functions to configure the logging behavior, such as setting the output file, enabling or disabling colors, and toggling timestamps and prefixes. The use of a separate worker thread ensures that logging operations do not block the main application flow, making it suitable for high-performance applications. The code is structured to be imported and used in other projects, providing a robust and customizable logging solution.
# Imports and Dependencies

---
- `log.h`
- `chrono`
- `condition_variable`
- `cstdarg`
- `cstdio`
- `mutex`
- `sstream`
- `thread`
- `vector`


# Global Variables

---
### common\_log\_verbosity\_thold
- **Type**: `int`
- **Description**: The variable `common_log_verbosity_thold` is an integer that represents the threshold for log verbosity levels. It is initialized with the value `LOG_DEFAULT_LLAMA`, which is likely a predefined constant representing a default verbosity level.
- **Use**: This variable is used to control the verbosity level of log messages, determining which messages are displayed based on their verbosity compared to this threshold.


---
### g\_col
- **Type**: `std::vector<const char *>`
- **Description**: The `g_col` variable is a static global vector of constant character pointers, initialized with empty strings. It is used to store ANSI color codes for different log levels, which can be enabled or disabled based on user preference.
- **Use**: This variable is used to apply color formatting to log messages, enhancing readability by distinguishing log levels with different colors.


# Data Structures

---
### common\_log\_col<!-- {{#data_structure:common_log_col}} -->
- **Type**: `enum`
- **Members**:
    - `COMMON_LOG_COL_DEFAULT`: Represents the default log color, with a value of 0.
    - `COMMON_LOG_COL_BOLD`: Represents a bold log color.
    - `COMMON_LOG_COL_RED`: Represents a red log color.
    - `COMMON_LOG_COL_GREEN`: Represents a green log color.
    - `COMMON_LOG_COL_YELLOW`: Represents a yellow log color.
    - `COMMON_LOG_COL_BLUE`: Represents a blue log color.
    - `COMMON_LOG_COL_MAGENTA`: Represents a magenta log color.
    - `COMMON_LOG_COL_CYAN`: Represents a cyan log color.
    - `COMMON_LOG_COL_WHITE`: Represents a white log color.
- **Description**: The `common_log_col` enum defines a set of constants representing different color codes used for logging purposes. Each enumerator corresponds to a specific color or style (such as bold) that can be applied to log messages to enhance readability and convey different levels of importance or types of information. The enum values are used to index into a color array, allowing for dynamic color configuration in the logging system.


---
### common\_log\_entry<!-- {{#data_structure:common_log_entry}} -->
- **Type**: `struct`
- **Members**:
    - `level`: An enumeration representing the log level of the entry.
    - `prefix`: A boolean indicating whether a prefix should be added to the log entry.
    - `timestamp`: A 64-bit integer representing the timestamp of the log entry in microseconds.
    - `msg`: A vector of characters containing the log message.
    - `is_end`: A boolean flag indicating if this entry signals the worker thread to stop.
- **Description**: The `common_log_entry` struct is a data structure used to represent a single log entry in a logging system. It contains information about the log level, whether a prefix should be added, the timestamp of the log entry, the actual log message, and a flag to signal the end of logging. This struct is part of a larger logging mechanism that manages log entries and outputs them to various destinations, such as standard output or a file, with optional color coding and formatting.
- **Member Functions**:
    - [`common_log_entry::print`](#common_log_entryprint)

**Methods**

---
#### common\_log\_entry::print<!-- {{#callable:common_log_entry::print}} -->
The `print` function outputs a log message to a specified file or standard output streams, with optional timestamp and color-coded log level prefixing.
- **Inputs**:
    - `file`: An optional FILE pointer to specify the output stream; defaults to nullptr, which means the function will choose between stdout and stderr based on log level.
- **Control Flow**:
    - Initialize `fcur` with the provided `file` argument.
    - If `fcur` is nullptr, determine the appropriate output stream based on the log level and verbosity threshold, defaulting to stdout or stderr.
    - If the log level is not NONE or CONT and prefixing is enabled, optionally prepend a timestamp and a color-coded log level indicator to the message.
    - Output the log message stored in `msg` to the determined output stream `fcur`.
    - If the log level is WARN, ERROR, or DEBUG, reset the color to default after the message.
    - Flush the output stream to ensure the message is written immediately.
- **Output**: The function does not return a value; it outputs the log message to the specified or determined output stream.
- **See also**: [`common_log_entry`](#common_log_entry)  (Data Structure)



---
### common\_log<!-- {{#data_structure:common_log}} -->
- **Type**: `struct`
- **Members**:
    - `mtx`: A mutex used to synchronize access to the log data.
    - `thrd`: A thread that processes log entries asynchronously.
    - `cv`: A condition variable used to manage thread synchronization.
    - `file`: A file pointer for logging output to a file.
    - `prefix`: A boolean indicating whether to include a prefix in log entries.
    - `timestamps`: A boolean indicating whether to include timestamps in log entries.
    - `running`: A boolean indicating whether the logging thread is active.
    - `t_start`: An integer representing the start time for timestamp calculations.
    - `entries`: A vector serving as a ring buffer for log entries.
    - `head`: An index pointing to the start of the ring buffer.
    - `tail`: An index pointing to the end of the ring buffer.
    - `cur`: A temporary storage for the current log entry being processed.
- **Description**: The `common_log` struct is a logging utility designed to handle log entries asynchronously using a ring buffer and a dedicated worker thread. It supports features such as file output, optional prefixes, and timestamps for log entries. The struct manages synchronization using mutexes and condition variables, ensuring thread-safe operations. It can dynamically expand its buffer to accommodate more log entries and provides methods to control logging behavior, such as pausing, resuming, and setting output file paths.
- **Member Functions**:
    - [`common_log::common_log`](#common_logcommon_log)
    - [`common_log::common_log`](#common_logcommon_log)
    - [`common_log::~common_log`](#common_logcommon_log)
    - [`common_log::add`](#common_logadd)
    - [`common_log::resume`](#common_logresume)
    - [`common_log::pause`](#common_logpause)
    - [`common_log::set_file`](#common_logset_file)
    - [`common_log::set_colors`](#common_logset_colors)
    - [`common_log::set_prefix`](#common_logset_prefix)
    - [`common_log::set_timestamps`](#common_logset_timestamps)

**Methods**

---
#### common\_log::common\_log<!-- {{#callable:common_log::common_log}} -->
The `common_log` constructor initializes a logging system with a default capacity of 256 entries, setting up necessary resources and starting the logging thread.
- **Inputs**: None
- **Control Flow**:
    - The constructor `common_log()` is called, which in turn calls the overloaded constructor `common_log(size_t capacity)` with a default capacity of 256.
    - In the `common_log(size_t capacity)` constructor, several member variables are initialized, including setting `file` to `nullptr`, `prefix`, `timestamps`, and `running` to `false`, and capturing the start time with `t_start = t_us()`.
    - The `entries` vector is resized to the specified capacity, and each entry's message buffer is initialized to a size of 256 characters.
    - The `head` and `tail` indices for the ring buffer are set to 0.
    - The `resume()` method is called to start the logging thread.
- **Output**: The function does not return any value; it initializes the `common_log` object.
- **See also**: [`common_log`](#common_log)  (Data Structure)


---
#### common\_log::common\_log<!-- {{#callable:common_log::common_log}} -->
The `common_log` constructor initializes a logging system with a specified capacity, setting up a ring buffer for log entries and starting a worker thread to process log messages.
- **Inputs**:
    - `capacity`: A `size_t` value representing the initial capacity of the log entries buffer.
- **Control Flow**:
    - Initialize member variables `file`, `prefix`, `timestamps`, and `running` to default values.
    - Set `t_start` to the current time in microseconds using `t_us()`.
    - Resize the `entries` vector to the specified `capacity`, initializing each entry's message buffer to 256 characters.
    - Set `head` and `tail` indices to 0, indicating the start of the ring buffer.
    - Call the `resume()` method to start the worker thread for processing log entries.
- **Output**: The constructor does not return a value; it initializes the `common_log` object.
- **Functions called**:
    - [`t_us`](#t_us)
    - [`common_log::resume`](#common_logresume)
- **See also**: [`common_log`](#common_log)  (Data Structure)


---
#### common\_log::\~common\_log<!-- {{#callable:common_log::~common_log}} -->
The destructor `~common_log` ensures that the logging process is paused and any open file is closed when a `common_log` object is destroyed.
- **Inputs**: None
- **Control Flow**:
    - Call the `pause()` method to stop the logging process and ensure the worker thread is properly terminated.
    - Check if the `file` pointer is not null, indicating an open file.
    - If a file is open, close it using `fclose(file)`.
- **Output**: This function does not return any value as it is a destructor.
- **Functions called**:
    - [`common_log::pause`](#common_logpause)
- **See also**: [`common_log`](#common_log)  (Data Structure)


---
#### common\_log::add<!-- {{#callable:common_log::add}} -->
The `add` function logs a formatted message into a thread-safe ring buffer, expanding the buffer if necessary, and notifies a worker thread for processing.
- **Inputs**:
    - `level`: An enumeration value of type `ggml_log_level` indicating the severity level of the log message.
    - `fmt`: A C-style string that contains the format string for the log message.
    - `args`: A `va_list` containing the arguments for the format string.
- **Control Flow**:
    - Acquire a lock on the mutex to ensure thread safety.
    - Check if the logging system is running; if not, return immediately without logging the message.
    - Retrieve the current log entry at the `tail` index of the ring buffer.
    - Copy the `va_list` to avoid using it twice, as it may be needed for buffer expansion.
    - Format the message using `vsnprintf` and store it in the current log entry's message buffer.
    - If the formatted message exceeds the current buffer size, resize the buffer and reformat the message.
    - Set the log entry's level, prefix, and timestamp based on the current settings.
    - Advance the `tail` index, wrapping around if necessary, and check if the buffer needs expansion.
    - If the buffer is full (tail meets head), double the buffer size and adjust head and tail indices.
    - Notify the condition variable to signal the worker thread that a new log entry is available.
- **Output**: The function does not return a value; it modifies the internal state of the `common_log` object by adding a new log entry to the buffer.
- **Functions called**:
    - [`t_us`](#t_us)
- **See also**: [`common_log`](#common_log)  (Data Structure)


---
#### common\_log::resume<!-- {{#callable:common_log::resume}} -->
The `resume` function starts a worker thread to process log entries if it is not already running.
- **Inputs**: None
- **Control Flow**:
    - Acquire a lock on the mutex `mtx` to ensure thread safety.
    - Check if the `running` flag is true; if so, return immediately as the thread is already running.
    - Set the `running` flag to true to indicate the thread is active.
    - Create a new thread `thrd` that continuously processes log entries from the `entries` vector.
    - Within the thread, acquire a unique lock on `mtx` and wait on the condition variable `cv` until there is a new entry to process (i.e., `head != tail`).
    - Retrieve the current log entry from `entries` at the `head` position and update `head` to the next position in the circular buffer.
    - If the current entry's `is_end` flag is true, break the loop to stop the thread.
    - Print the current log entry to standard output and, if a file is specified, also to the file.
- **Output**: The function does not return any value.
- **See also**: [`common_log`](#common_log)  (Data Structure)


---
#### common\_log::pause<!-- {{#callable:common_log::pause}} -->
The `pause` function stops the logging process by signaling the worker thread to terminate and waits for it to join.
- **Inputs**: None
- **Control Flow**:
    - Acquire a lock on the mutex `mtx` to ensure thread safety.
    - Check if the logging process is already stopped by evaluating the `running` flag; if not running, return immediately.
    - Set the `running` flag to false to indicate the logging process should stop.
    - Create a special log entry with `is_end` set to true to signal the worker thread to terminate.
    - Update the `tail` index to point to the next position in the ring buffer.
    - Notify the worker thread waiting on the condition variable `cv` that a new entry is available.
    - Wait for the worker thread to finish execution and join it using `thrd.join()`.
- **Output**: The function does not return any value.
- **See also**: [`common_log`](#common_log)  (Data Structure)


---
#### common\_log::set\_file<!-- {{#callable:common_log::set_file}} -->
The `set_file` function sets a new file for logging by pausing the logging process, closing any existing file, opening a new file if a path is provided, and then resuming the logging process.
- **Inputs**:
    - `path`: A constant character pointer representing the file path to open for logging; if null, no file will be opened.
- **Control Flow**:
    - The function begins by calling `pause()` to stop the logging process safely.
    - It checks if there is an existing file open (`file` is not null) and closes it using `fclose(file)`.
    - If the `path` is not null, it attempts to open the file at the given path in write mode (`fopen(path, "w")`) and assigns it to `file`.
    - If the `path` is null, it sets `file` to `nullptr`, indicating no file is open for logging.
    - Finally, it calls `resume()` to restart the logging process.
- **Output**: The function does not return any value.
- **Functions called**:
    - [`common_log::pause`](#common_logpause)
    - [`common_log::resume`](#common_logresume)
- **See also**: [`common_log`](#common_log)  (Data Structure)


---
#### common\_log::set\_colors<!-- {{#callable:common_log::set_colors}} -->
The `set_colors` function configures the global color settings for log messages based on the input boolean flag.
- **Inputs**:
    - `colors`: A boolean flag indicating whether to enable (true) or disable (false) color settings for log messages.
- **Control Flow**:
    - The function begins by pausing the logging process using the `pause()` method.
    - It checks the value of the `colors` parameter.
    - If `colors` is true, it assigns predefined color codes to the global color array `g_col` for various log levels.
    - If `colors` is false, it iterates over the `g_col` array and sets each element to an empty string, effectively disabling colors.
    - Finally, the function resumes the logging process using the `resume()` method.
- **Output**: The function does not return any value; it modifies the global color settings for log messages.
- **Functions called**:
    - [`common_log::pause`](#common_logpause)
    - [`common_log::resume`](#common_logresume)
- **See also**: [`common_log`](#common_log)  (Data Structure)


---
#### common\_log::set\_prefix<!-- {{#callable:common_log::set_prefix}} -->
The `set_prefix` function sets the `prefix` flag in the `common_log` structure to control whether log entries should include a prefix.
- **Inputs**:
    - `prefix`: A boolean value indicating whether to enable or disable the prefix for log entries.
- **Control Flow**:
    - Acquire a lock on the mutex `mtx` to ensure thread safety.
    - Set the `prefix` member variable of the `common_log` instance to the value of the `prefix` parameter.
- **Output**: This function does not return any value.
- **See also**: [`common_log`](#common_log)  (Data Structure)


---
#### common\_log::set\_timestamps<!-- {{#callable:common_log::set_timestamps}} -->
The `set_timestamps` function sets the `timestamps` flag in the `common_log` structure to enable or disable timestamping of log entries.
- **Inputs**:
    - `timestamps`: A boolean value indicating whether timestamps should be enabled (true) or disabled (false) for log entries.
- **Control Flow**:
    - Acquire a lock on the mutex `mtx` to ensure thread safety.
    - Set the `timestamps` member variable of the `common_log` instance to the provided boolean value.
- **Output**: This function does not return any value.
- **See also**: [`common_log`](#common_log)  (Data Structure)



# Functions

---
### common\_log\_set\_verbosity\_thold<!-- {{#callable:common_log_set_verbosity_thold}} -->
The function `common_log_set_verbosity_thold` sets the global logging verbosity threshold to a specified level.
- **Inputs**:
    - `verbosity`: An integer representing the desired verbosity level to set for logging.
- **Control Flow**:
    - The function assigns the input `verbosity` value to the global variable `common_log_verbosity_thold`.
- **Output**: This function does not return any value.


---
### t\_us<!-- {{#callable:t_us}} -->
The `t_us` function returns the current time in microseconds since the Unix epoch.
- **Inputs**: None
- **Control Flow**:
    - The function calls `std::chrono::system_clock::now()` to get the current time point.
    - It then calculates the duration since the epoch using `time_since_epoch()`.
    - The duration is cast to microseconds using `std::chrono::duration_cast<std::chrono::microseconds>()`.
    - Finally, it returns the count of microseconds as an `int64_t`.
- **Output**: The function returns an `int64_t` representing the current time in microseconds since the Unix epoch.


---
### common\_log\_init<!-- {{#callable:common_log_init}} -->
The `common_log_init` function initializes and returns a new instance of the `common_log` structure.
- **Inputs**: None
- **Control Flow**:
    - The function simply returns a new instance of the `common_log` structure using the `new` operator.
- **Output**: A pointer to a newly allocated `common_log` structure.


---
### common\_log\_main<!-- {{#callable:common_log_main}} -->
The `common_log_main` function returns a pointer to a static instance of the `common_log` structure.
- **Inputs**: None
- **Control Flow**:
    - Declare a static `common_log` structure named `log`.
    - Return the address of the `log` structure.
- **Output**: A pointer to a static `common_log` structure.


---
### common\_log\_pause<!-- {{#callable:common_log_pause}} -->
The `common_log_pause` function pauses the logging process by invoking the `pause` method on a `common_log` object.
- **Inputs**:
    - `log`: A pointer to a `common_log` structure, which represents the logging system to be paused.
- **Control Flow**:
    - The function calls the `pause` method on the `common_log` object pointed to by `log`.
- **Output**: This function does not return any value.


---
### common\_log\_resume<!-- {{#callable:common_log_resume}} -->
The `common_log_resume` function resumes the logging process by invoking the `resume` method on a `common_log` object.
- **Inputs**:
    - `log`: A pointer to a `common_log` structure, which represents the logging system to be resumed.
- **Control Flow**:
    - The function calls the `resume` method on the `common_log` object pointed to by `log`.
- **Output**: This function does not return any value.


---
### common\_log\_free<!-- {{#callable:common_log_free}} -->
The `common_log_free` function deallocates memory for a `common_log` structure.
- **Inputs**:
    - `log`: A pointer to a `common_log` structure that needs to be deallocated.
- **Control Flow**:
    - The function takes a pointer to a `common_log` structure as an argument.
    - It uses the `delete` operator to deallocate the memory associated with the `common_log` structure.
- **Output**: The function does not return any value.


---
### common\_log\_add<!-- {{#callable:common_log_add}} -->
The `common_log_add` function adds a formatted log entry to a `common_log` structure at a specified log level.
- **Inputs**:
    - `log`: A pointer to a `common_log` structure where the log entry will be added.
    - `level`: An enumeration value of type `ggml_log_level` indicating the severity level of the log entry.
    - `fmt`: A C-style string containing the format for the log message, similar to `printf`.
    - `...`: A variable number of arguments that correspond to the format specifiers in `fmt`.
- **Control Flow**:
    - Initialize a `va_list` variable `args` to handle the variable arguments.
    - Start the variable argument list processing with `va_start`, using `fmt` as the last fixed argument.
    - Call the `add` method of the `common_log` structure, passing the log level, format string, and the `va_list` `args`.
    - End the variable argument list processing with `va_end`.
- **Output**: This function does not return a value; it modifies the `common_log` structure by adding a new log entry.


---
### common\_log\_set\_file<!-- {{#callable:common_log_set_file}} -->
The `common_log_set_file` function sets the file path for logging output in a `common_log` structure.
- **Inputs**:
    - `log`: A pointer to a `common_log` structure where the log file path will be set.
    - `file`: A constant character pointer representing the file path to which log entries will be written.
- **Control Flow**:
    - The function calls the `set_file` method on the `common_log` structure pointed to by `log`, passing the `file` argument to it.
- **Output**: This function does not return any value.


---
### common\_log\_set\_colors<!-- {{#callable:common_log_set_colors}} -->
The `common_log_set_colors` function configures whether log messages should be displayed with colors.
- **Inputs**:
    - `log`: A pointer to a `common_log` structure, which represents the logging system to be configured.
    - `colors`: A boolean value indicating whether colors should be enabled (`true`) or disabled (`false`) for log messages.
- **Control Flow**:
    - The function calls the `set_colors` method on the `common_log` object pointed to by `log`, passing the `colors` boolean as an argument.
- **Output**: This function does not return any value.


---
### common\_log\_set\_prefix<!-- {{#callable:common_log_set_prefix}} -->
The `common_log_set_prefix` function sets the prefix flag for a given `common_log` instance, determining whether log entries should include a prefix.
- **Inputs**:
    - `log`: A pointer to a `common_log` structure, representing the logging instance whose prefix setting is to be modified.
    - `prefix`: A boolean value indicating whether the prefix should be enabled (`true`) or disabled (`false`).
- **Control Flow**:
    - The function directly calls the `set_prefix` method on the `common_log` instance pointed to by `log`, passing the `prefix` boolean value as an argument.
- **Output**: This function does not return any value.


---
### common\_log\_set\_timestamps<!-- {{#callable:common_log_set_timestamps}} -->
The function `common_log_set_timestamps` configures a `common_log` object to enable or disable timestamping of log entries.
- **Inputs**:
    - `log`: A pointer to a `common_log` structure that represents the logging system to be configured.
    - `timestamps`: A boolean value indicating whether timestamps should be enabled (`true`) or disabled (`false`) for log entries.
- **Control Flow**:
    - The function calls the `set_timestamps` method on the `common_log` object pointed to by `log`, passing the `timestamps` boolean value as an argument.
- **Output**: This function does not return any value.


