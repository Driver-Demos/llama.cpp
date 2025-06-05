# Purpose
This C++ source code file provides a comprehensive set of utilities and functions primarily focused on CPU management, string manipulation, file system operations, and model utilities. The file includes functions to determine the number of physical and mathematical CPUs available on a system, which is crucial for optimizing multi-threaded applications. It also includes platform-specific code to handle different operating systems like Linux, macOS, and Windows, ensuring compatibility across various environments.

Additionally, the file contains a variety of string utility functions for formatting, splitting, joining, and manipulating strings, which are essential for handling text data efficiently. The file also includes functions for parsing command-line arguments related to CPU parameters, setting process priorities, and managing file system operations such as creating directories and validating filenames. Furthermore, it provides utilities for handling model-related tasks, such as loading models, tokenizing text, and managing control vectors, which are critical for applications involving machine learning models. Overall, this file serves as a utility library that can be integrated into larger projects to provide essential system and string handling capabilities.
# Imports and Dependencies

---
- `ggml.h`
- `gguf.h`
- `common.h`
- `log.h`
- `llama.h`
- `algorithm`
- `cinttypes`
- `climits`
- `cmath`
- `codecvt`
- `cstdarg`
- `cstring`
- `ctime`
- `filesystem`
- `fstream`
- `iostream`
- `iterator`
- `regex`
- `sstream`
- `string`
- `thread`
- `unordered_map`
- `unordered_set`
- `vector`
- `sys/types.h`
- `sys/sysctl.h`
- `locale`
- `windows.h`
- `fcntl.h`
- `io.h`
- `sys/ioctl.h`
- `sys/stat.h`
- `unistd.h`
- `pthread.h`
- `sys/resource.h`


# Functions

---
### cpu\_get\_num\_physical\_cores<!-- {{#callable:cpu_get_num_physical_cores}} -->
The function `cpu_get_num_physical_cores` determines the number of physical CPU cores available on the system, with different implementations for Linux, macOS, and Windows.
- **Inputs**: None
- **Control Flow**:
    - Check if the platform is Linux, macOS, or Windows and execute the corresponding block of code.
    - For Linux, iterate over possible CPU indices, reading the 'thread_siblings' file to count unique sibling sets, which represent physical cores.
    - For macOS, use the `sysctlbyname` function to query the number of physical CPUs, first trying 'hw.perflevel0.physicalcpu' and then 'hw.physicalcpu'.
    - For Windows, use `GetLogicalProcessorInformationEx` to retrieve processor information and count the number of physical cores based on processor group count.
    - If none of the platform-specific methods succeed, fall back to using `std::thread::hardware_concurrency` to estimate the number of cores, adjusting for systems with more than 4 threads.
- **Output**: Returns an `int32_t` representing the number of physical CPU cores, or an estimated value if the exact number cannot be determined.


---
### cpuid<!-- {{#callable:cpuid}} -->
The `cpuid` function executes the CPUID instruction to retrieve information about the CPU based on the specified leaf and subleaf values.
- **Inputs**:
    - `leaf`: An unsigned integer specifying the main function of the CPUID instruction to execute.
    - `subleaf`: An unsigned integer specifying the sub-function of the CPUID instruction to execute.
    - `eax`: A pointer to an unsigned integer where the result of the CPUID instruction's EAX register will be stored.
    - `ebx`: A pointer to an unsigned integer where the result of the CPUID instruction's EBX register will be stored.
    - `ecx`: A pointer to an unsigned integer where the result of the CPUID instruction's ECX register will be stored.
    - `edx`: A pointer to an unsigned integer where the result of the CPUID instruction's EDX register will be stored.
- **Control Flow**:
    - The function uses inline assembly to execute the CPUID instruction.
    - The `movq` instruction saves the value of the RBX register to RSI to preserve it across the CPUID call.
    - The `cpuid` instruction is executed with the specified leaf and subleaf values, and the results are stored in the EAX, EBX, ECX, and EDX registers.
    - The `xchgq` instruction swaps the values of the RBX and RSI registers to restore the original RBX value.
- **Output**: The function does not return a value, but it modifies the values pointed to by the `eax`, `ebx`, `ecx`, and `edx` pointers with the results of the CPUID instruction.


---
### pin\_cpu<!-- {{#callable:pin_cpu}} -->
The `pin_cpu` function sets the CPU affinity of the current thread to a specific CPU core.
- **Inputs**:
    - `cpu`: An integer representing the CPU core to which the current thread should be pinned.
- **Control Flow**:
    - Initialize a `cpu_set_t` variable named `mask`.
    - Clear the CPU set `mask` using `CPU_ZERO`.
    - Add the specified CPU core to the set `mask` using `CPU_SET`.
    - Set the CPU affinity of the current thread to the specified CPU core using `pthread_setaffinity_np`.
- **Output**: Returns an integer indicating the success or failure of setting the CPU affinity, where 0 typically indicates success and a non-zero value indicates an error.


---
### is\_hybrid\_cpu<!-- {{#callable:is_hybrid_cpu}} -->
The `is_hybrid_cpu` function checks if the CPU is a hybrid architecture by using the CPUID instruction to examine specific CPU features.
- **Inputs**: None
- **Control Flow**:
    - Declare four unsigned integer variables: eax, ebx, ecx, and edx.
    - Call the [`cpuid`](#cpuid) function with leaf 7 and subleaf 0 to populate the eax, ebx, ecx, and edx registers with CPU feature information.
    - Check if the 15th bit of the edx register is set, indicating a hybrid CPU architecture.
    - Return true if the 15th bit is set, otherwise return false.
- **Output**: A boolean value indicating whether the CPU is a hybrid architecture (true if it is, false otherwise).
- **Functions called**:
    - [`cpuid`](#cpuid)


---
### is\_running\_on\_efficiency\_core<!-- {{#callable:is_running_on_efficiency_core}} -->
The function `is_running_on_efficiency_core` checks if the current CPU core is an Intel Atom efficiency core.
- **Inputs**: None
- **Control Flow**:
    - Declare four unsigned integers: eax, ebx, ecx, and edx.
    - Call the [`cpuid`](#cpuid) function with leaf 0x1a and subleaf 0 to populate eax, ebx, ecx, and edx with CPU information.
    - Define an integer `intel_atom` with the value 0x20, representing the identifier for Intel Atom efficiency cores.
    - Extract the core type from the `eax` register by applying a bitmask and right-shifting to isolate the relevant bits.
    - Return true if the extracted core type matches `intel_atom`, indicating the current core is an efficiency core; otherwise, return false.
- **Output**: A boolean value indicating whether the current CPU core is an Intel Atom efficiency core.
- **Functions called**:
    - [`cpuid`](#cpuid)


---
### cpu\_count\_math\_cpus<!-- {{#callable:cpu_count_math_cpus}} -->
The `cpu_count_math_cpus` function calculates the number of CPUs that are suitable for mathematical computations, excluding efficiency cores and hyperthreaded cores.
- **Inputs**:
    - `n_cpu`: The total number of CPUs available on the system.
- **Control Flow**:
    - Initialize a result counter to zero.
    - Iterate over each CPU index from 0 to n_cpu.
    - Attempt to pin the current CPU; if unsuccessful, return -1.
    - Check if the current CPU is an efficiency core; if so, skip to the next iteration.
    - Skip the next CPU index to avoid hyperthreading, as it is not useful for linear algebra.
    - Increment the result counter for each suitable CPU.
    - Return the result counter, which represents the number of suitable CPUs.
- **Output**: The function returns an integer representing the number of CPUs suitable for mathematical computations, or -1 if pinning any CPU fails.
- **Functions called**:
    - [`pin_cpu`](#pin_cpu)
    - [`is_running_on_efficiency_core`](#is_running_on_efficiency_core)


---
### cpu\_get\_num\_math<!-- {{#callable:cpu_get_num_math}} -->
The `cpu_get_num_math` function determines the number of CPUs available for mathematical computations on a system, particularly focusing on Linux systems with x86_64 architecture.
- **Inputs**: None
- **Control Flow**:
    - Check if the system is x86_64, Linux, and not Android.
    - Use `sysconf` to get the number of online processors (`n_cpu`).
    - If `n_cpu` is less than 1, return the number of physical cores using `cpu_get_num_physical_cores()`.
    - Check if the CPU is a hybrid CPU using `is_hybrid_cpu()`.
    - If the CPU is hybrid, get the current thread's CPU affinity and count the math-capable CPUs using `cpu_count_math_cpus()`.
    - Restore the original CPU affinity after counting.
    - If a valid number of math CPUs is found, return it.
    - If any condition fails, return the number of physical cores.
- **Output**: Returns an `int32_t` representing the number of CPUs suitable for mathematical operations.
- **Functions called**:
    - [`cpu_get_num_physical_cores`](#cpu_get_num_physical_cores)
    - [`is_hybrid_cpu`](#is_hybrid_cpu)
    - [`cpu_count_math_cpus`](#cpu_count_math_cpus)


---
### set\_process\_priority<!-- {{#callable:set_process_priority}} -->
The `set_process_priority` function sets the process priority based on the given scheduling priority level.
- **Inputs**:
    - `prio`: An enumeration value of type `ggml_sched_priority` representing the desired scheduling priority level for the process.
- **Control Flow**:
    - Check if the priority is `GGML_SCHED_PRIO_NORMAL`; if so, return `true` immediately as no change is needed.
    - Initialize an integer `p` to 0, which will hold the priority value to be set.
    - Use a switch statement to map the `prio` enumeration to a corresponding integer priority value `p`.
    - Call `setpriority` with `PRIO_PROCESS`, `0`, and `p` to set the process priority.
    - If `setpriority` fails, log a warning message with the priority and error details, and return `false`.
    - If `setpriority` succeeds, return `true`.
- **Output**: A boolean value indicating whether the process priority was successfully set (`true`) or not (`false`).


---
### postprocess\_cpu\_params<!-- {{#callable:postprocess_cpu_params}} -->
The `postprocess_cpu_params` function adjusts and validates the CPU parameters for threading based on a role model or system defaults.
- **Inputs**:
    - `cpuparams`: A reference to a `cpu_params` object that holds the CPU parameters to be processed.
    - `role_model`: A pointer to a `cpu_params` object that serves as a template for default values if `cpuparams` is invalid.
- **Control Flow**:
    - Initialize `n_set` to 0 to count the number of set bits in the CPU mask.
    - Check if `cpuparams.n_threads` is less than 0, indicating invalid parameters.
    - If invalid, and `role_model` is provided, copy `role_model` into `cpuparams`; otherwise, set `cpuparams.n_threads` to the number of math-capable CPUs.
    - Iterate over the CPU mask array up to `GGML_MAX_N_THREADS`, incrementing `n_set` for each set bit.
    - If `n_set` is non-zero but less than `cpuparams.n_threads`, log a warning about potential performance issues due to insufficient set bits.
- **Output**: The function does not return a value; it modifies the `cpuparams` object in place.
- **Functions called**:
    - [`cpu_get_num_math`](#cpu_get_num_math)


---
### parse\_cpu\_range<!-- {{#callable:parse_cpu_range}} -->
The `parse_cpu_range` function parses a string representing a range of CPU indices and sets the corresponding indices in a boolean mask to true.
- **Inputs**:
    - `range`: A string representing a range of CPU indices in the format '[<start>]-[<end>]'.
    - `boolmask`: A boolean array of size GGML_MAX_N_THREADS, which will be updated to reflect the parsed CPU range.
- **Control Flow**:
    - Find the position of the dash '-' in the input string `range` to separate the start and end indices.
    - If no dash is found, log an error and return false, indicating an invalid format.
    - Determine the start index `start_i` from the substring before the dash; if the dash is at the start, set `start_i` to 0.
    - Check if `start_i` is within bounds (less than GGML_MAX_N_THREADS); if not, log an error and return false.
    - Determine the end index `end_i` from the substring after the dash; if the dash is at the end, set `end_i` to GGML_MAX_N_THREADS - 1.
    - Check if `end_i` is within bounds (less than GGML_MAX_N_THREADS); if not, log an error and return false.
    - Iterate from `start_i` to `end_i`, setting each corresponding index in `boolmask` to true.
    - Return true to indicate successful parsing and updating of the boolean mask.
- **Output**: A boolean value indicating whether the parsing and updating of the boolean mask was successful.


---
### parse\_cpu\_mask<!-- {{#callable:parse_cpu_mask}} -->
The `parse_cpu_mask` function parses a hexadecimal CPU mask string and updates a boolean array to represent the active CPUs.
- **Inputs**:
    - `mask`: A string representing a hexadecimal CPU mask, potentially prefixed with '0x'.
    - `boolmask`: A reference to a boolean array of size `GGML_MAX_N_THREADS` that will be updated to reflect the active CPUs based on the mask.
- **Control Flow**:
    - Check if the mask string has a '0x' prefix and adjust the starting index accordingly.
    - Calculate the number of hexadecimal digits to process, limiting to a maximum of 128 digits.
    - Iterate over each character in the mask string, converting it from hexadecimal to a numeric value.
    - For each character, update the corresponding bits in the `boolmask` array to reflect the active CPUs.
    - If an invalid character is encountered, log an error and return false.
    - If all characters are valid, return true.
- **Output**: Returns a boolean value indicating whether the parsing was successful (true) or if an error occurred (false).


---
### common\_init<!-- {{#callable:common_init}} -->
The `common_init` function initializes logging for the application and logs build information.
- **Inputs**: None
- **Control Flow**:
    - Sets a custom logging function using `llama_log_set` that logs messages if the log level is below a certain threshold.
    - Determines the build type (debug or release) based on the `NDEBUG` macro.
    - Logs the build number, commit, compiler, build target, and build type using `LOG_INF`.
- **Output**: The function does not return any value.
- **Functions called**:
    - [`common_log_add`](log.cpp.driver.md#common_log_add)
    - [`common_log_main`](log.cpp.driver.md#common_log_main)


---
### common\_params\_get\_system\_info<!-- {{#callable:common_params_get_system_info}} -->
The function `common_params_get_system_info` generates a string containing system information related to CPU threads and system capabilities.
- **Inputs**:
    - `params`: A `common_params` structure containing CPU parameters, including the number of threads and batch thread settings.
- **Control Flow**:
    - Initialize an output string stream `os`.
    - Append the number of threads from `params.cpuparams.n_threads` to `os`.
    - Check if `params.cpuparams_batch.n_threads` is not -1, and if so, append the batch thread count to `os`.
    - On Windows 7 and later (excluding MinGW64), retrieve the logical processor count using `GetActiveProcessorCount` and append it to `os`.
    - On other systems, append the hardware concurrency value from `std::thread::hardware_concurrency` to `os`.
    - Append the result of `llama_print_system_info()` to `os`.
    - Return the constructed string from `os`.
- **Output**: A `std::string` containing formatted system information about CPU threads and system capabilities.


---
### string\_format<!-- {{#callable:string_format}} -->
The `string_format` function formats a string using a printf-style format string and variable arguments, returning the formatted result as a `std::string`.
- **Inputs**:
    - `fmt`: A C-style string that specifies the format of the output string, similar to the format string used in printf.
    - `...`: A variable number of arguments that are formatted according to the format specifiers in `fmt`.
- **Control Flow**:
    - Initialize a `va_list` named `ap` and start it with `va_start`, using `fmt` as the last fixed argument.
    - Create a copy of `ap` named `ap2` using `va_copy`.
    - Use `vsnprintf` with a `NULL` buffer to calculate the size needed for the formatted string, storing the result in `size`.
    - Assert that `size` is non-negative and less than `INT_MAX`.
    - Create a `std::vector<char>` buffer of size `size + 1` to hold the formatted string.
    - Use `vsnprintf` again with the buffer to format the string, storing the result in `size2`.
    - Assert that `size2` is equal to `size`, ensuring the buffer was correctly sized.
    - End the `va_list` copies `ap2` and `ap` using `va_end`.
    - Return a `std::string` constructed from the buffer data, excluding the null terminator.
- **Output**: A `std::string` containing the formatted output.


---
### string\_strip<!-- {{#callable:string_strip}} -->
The `string_strip` function removes leading and trailing whitespace from a given string.
- **Inputs**:
    - `str`: A constant reference to a `std::string` object from which leading and trailing whitespace will be removed.
- **Control Flow**:
    - Initialize `start` to 0 and `end` to the size of the input string `str`.
    - Increment `start` while it points to a whitespace character and is less than `end`.
    - Decrement `end` while it points to a whitespace character and is greater than `start`.
    - Return a substring of `str` from `start` to `end` (exclusive).
- **Output**: A `std::string` that is a copy of the input string `str` with leading and trailing whitespace removed.


---
### string\_get\_sortable\_timestamp<!-- {{#callable:string_get_sortable_timestamp}} -->
The function `string_get_sortable_timestamp` generates a current timestamp in a sortable string format with nanosecond precision.
- **Inputs**: None
- **Control Flow**:
    - The function uses `std::chrono::system_clock` to get the current time as a `time_point`.
    - It converts the `time_point` to a `time_t` object to format the date and time up to seconds using `std::strftime`.
    - The function calculates the nanoseconds part by taking the remainder of the current time since epoch divided by one billion and converts it to a string using `snprintf`.
    - It concatenates the formatted date-time string and the nanoseconds string with a dot separator to form the final timestamp string.
- **Output**: A `std::string` representing the current timestamp in the format `YYYY_MM_DD-HH_MM_SS.nnnnnnnnn`, where `nnnnnnnnn` is the nanoseconds part.


---
### string\_replace\_all<!-- {{#callable:string_replace_all}} -->
The `string_replace_all` function replaces all occurrences of a specified substring within a given string with another substring.
- **Inputs**:
    - `s`: A reference to the string in which occurrences of the search substring will be replaced.
    - `search`: The substring to search for within the string `s`.
    - `replace`: The substring to replace each occurrence of the search substring with.
- **Control Flow**:
    - Check if the `search` string is empty; if so, return immediately as there's nothing to replace.
    - Initialize a `builder` string with reserved space equal to the length of `s` to optimize memory allocation.
    - Use a loop to find each occurrence of `search` in `s` starting from `last_pos`.
    - For each occurrence found, append the portion of `s` from `last_pos` to the found position to `builder`, followed by the `replace` string.
    - Update `last_pos` to the position after the found occurrence of `search`.
    - After the loop, append the remaining part of `s` from `last_pos` to the end to `builder`.
    - Move the contents of `builder` back to `s` to complete the replacement.
- **Output**: The function modifies the input string `s` in place, replacing all occurrences of `search` with `replace`.


---
### string\_ends\_with<!-- {{#callable:string_ends_with}} -->
The `string_ends_with` function checks if a given string ends with a specified suffix.
- **Inputs**:
    - `str`: A `std::string_view` representing the string to be checked.
    - `suffix`: A `std::string_view` representing the suffix to check against the end of the string.
- **Control Flow**:
    - Check if the size of `str` is greater than or equal to the size of `suffix`.
    - If true, compare the substring of `str` that is the same length as `suffix` starting from the end of `str` with `suffix`.
    - Return true if the comparison is equal, otherwise return false.
- **Output**: A boolean value indicating whether the string ends with the specified suffix.


---
### string\_find\_partial\_stop<!-- {{#callable:string_find_partial_stop}} -->
The `string_find_partial_stop` function searches for the longest suffix of the `stop` string that matches the end of the `str` string and returns the starting index of this suffix in `str`.
- **Inputs**:
    - `str`: A `std::string_view` representing the string in which to search for a partial match of the `stop` string.
    - `stop`: A `std::string_view` representing the string whose suffixes are checked against the end of `str`.
- **Control Flow**:
    - Check if both `str` and `stop` are not empty.
    - Retrieve the last character of `str`.
    - Iterate over the `stop` string from the last character to the first.
    - For each character in `stop`, check if it matches the last character of `str`.
    - If a match is found, extract the substring from the start of `stop` to the current character index.
    - Check if `str` ends with this substring using [`string_ends_with`](#string_ends_with).
    - If `str` ends with the substring, return the starting index of this substring in `str`.
    - If no match is found, return `std::string::npos`.
- **Output**: Returns the starting index of the longest suffix of `stop` that matches the end of `str`, or `std::string::npos` if no such suffix exists.
- **Functions called**:
    - [`string_ends_with`](#string_ends_with)


---
### regex\_escape<!-- {{#callable:regex_escape}} -->
The `regex_escape` function escapes special characters in a string to make it safe for use in a regular expression.
- **Inputs**:
    - `s`: A constant reference to a `std::string` that contains the input string which may have special characters that need to be escaped.
- **Control Flow**:
    - Define a static `std::regex` object named `special_chars` that matches any special character used in regular expressions, such as '.', '^', '$', '|', '(', ')', '*', '+', '?', '[', ']', '{', '}', and '\'.
    - Use `std::regex_replace` to replace each occurrence of a special character in the input string `s` with an escaped version of itself, i.e., prepend each special character with a backslash ('\').
    - Return the modified string with all special characters escaped.
- **Output**: A `std::string` with all special characters in the input string escaped, making it safe for use in regular expressions.


---
### string\_join<!-- {{#callable:string_join}} -->
The `string_join` function concatenates a vector of strings into a single string, with a specified separator between each element.
- **Inputs**:
    - `values`: A constant reference to a vector of strings that are to be joined.
    - `separator`: A constant reference to a string that will be used as the separator between each element in the vector.
- **Control Flow**:
    - Initialize an output string stream `result`.
    - Iterate over each string in the `values` vector using an index `i`.
    - For each string, if `i` is greater than 0, append the `separator` to the `result` stream.
    - Append the current string from `values` to the `result` stream.
- **Output**: Returns a single string that is the result of concatenating all the strings in `values`, separated by `separator`.


---
### string\_split<!-- {{#callable:string_split}} -->
The `string_split` function splits a given string into a vector of substrings based on a specified delimiter.
- **Inputs**:
    - `str`: The input string that needs to be split into parts.
    - `delimiter`: The string used as a delimiter to determine where the input string should be split.
- **Control Flow**:
    - Initialize an empty vector `parts` to store the resulting substrings.
    - Set `start` to 0 and find the first occurrence of `delimiter` in `str`, storing the position in `end`.
    - Enter a loop that continues as long as `end` is not `std::string::npos` (indicating the delimiter was found).
    - In each iteration, extract the substring from `start` to `end` and add it to `parts`.
    - Update `start` to the position just after the current `delimiter` and search for the next occurrence of `delimiter` starting from `start`.
    - After the loop, add the remaining part of the string (from `start` to the end of `str`) to `parts`.
- **Output**: A vector of strings, where each element is a substring of the input string `str` split by the `delimiter`.


---
### string\_repeat<!-- {{#callable:string_repeat}} -->
The `string_repeat` function creates a new string by repeating a given string a specified number of times.
- **Inputs**:
    - `str`: A constant reference to the input string that needs to be repeated.
    - `n`: A size_t value representing the number of times the input string should be repeated.
- **Control Flow**:
    - Check if the repeat count `n` is zero; if so, return an empty string immediately.
    - Reserve memory in the result string to optimize performance by pre-allocating space for the repeated string.
    - Iterate `n` times, appending the input string `str` to the result string in each iteration.
    - Return the result string containing the repeated sequence.
- **Output**: A new string that consists of the input string repeated `n` times, or an empty string if `n` is zero.


---
### string\_from<!-- {{#callable:string_from}} -->
The `string_from` function converts a `llama_batch` structure into a formatted string representation, detailing each token's properties.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure, which provides context for token processing.
    - `batch`: A reference to a `llama_batch` structure containing tokens and their associated metadata.
- **Control Flow**:
    - Initialize a stringstream `buf` and start the string with '[ '.
    - Set a boolean `first` to true to track the first token for formatting purposes.
    - Iterate over each token in the `batch` using a for loop.
    - For each token, check if it is the first; if not, append ', ' to `buf`, otherwise set `first` to false.
    - Call `common_token_to_piece` to convert the token to a string and remove non-printable characters from the result.
    - Append the token's index, detokenized string, position, sequence ID count, first sequence ID, and logits to `buf`.
    - Close the string with ' ]' and return the resulting string.
- **Output**: A formatted string representing the tokens in the `llama_batch`, including their indices, detokenized values, positions, sequence IDs, and logits.


---
### string\_process\_escapes<!-- {{#callable:string_process_escapes}} -->
The function `string_process_escapes` processes escape sequences in a given string, replacing them with their corresponding characters.
- **Inputs**:
    - `input`: A reference to a `std::string` that contains the input string to be processed for escape sequences.
- **Control Flow**:
    - Initialize `input_len` with the length of the input string and `output_idx` to 0.
    - Iterate over each character in the input string using `input_idx`.
    - Check if the current character is a backslash '\' and if there is a subsequent character.
    - If a valid escape sequence is found, replace it with the corresponding character (e.g., '\n' becomes newline, '\t' becomes tab, etc.).
    - For hexadecimal escape sequences starting with '\x', attempt to convert the following two characters to a hexadecimal value and replace the sequence with the corresponding character.
    - If no valid escape sequence is found, copy the backslash and the following character as is.
    - After processing, resize the input string to the new length indicated by `output_idx`.
- **Output**: The function modifies the input string in place, replacing escape sequences with their corresponding characters, and resizes the string to the new length.


---
### string\_parse\_kv\_override<!-- {{#callable:string_parse_kv_override}} -->
The function `string_parse_kv_override` parses a key-value string and appends a `llama_model_kv_override` object to a vector if the string is valid.
- **Inputs**:
    - `data`: A C-style string containing the key-value pair to be parsed, formatted as 'key=type:value'.
    - `overrides`: A reference to a vector of `llama_model_kv_override` objects where the parsed override will be appended if valid.
- **Control Flow**:
    - The function searches for the '=' character in the input string to separate the key from the value.
    - If the '=' character is not found or the key length is 128 or more, an error is logged and the function returns false.
    - A `llama_model_kv_override` object is initialized, and the key is copied from the input string up to the '=' character.
    - The function checks the type of the value by comparing the prefix after '=' with 'int:', 'float:', 'bool:', or 'str:'.
    - For 'int:', the function parses the integer value and assigns it to `val_i64` with the tag `LLAMA_KV_OVERRIDE_TYPE_INT`.
    - For 'float:', the function parses the float value and assigns it to `val_f64` with the tag `LLAMA_KV_OVERRIDE_TYPE_FLOAT`.
    - For 'bool:', the function checks if the value is 'true' or 'false', assigns the corresponding boolean to `val_bool`, and sets the tag `LLAMA_KV_OVERRIDE_TYPE_BOOL`.
    - For 'str:', the function checks if the string length is within 127 characters, copies it to `val_str`, and sets the tag `LLAMA_KV_OVERRIDE_TYPE_STR`.
    - If the type is invalid or the boolean value is not 'true' or 'false', an error is logged and the function returns false.
    - If the parsing is successful, the `llama_model_kv_override` object is appended to the `overrides` vector and the function returns true.
- **Output**: Returns a boolean indicating whether the parsing and appending of the key-value override was successful.


---
### fs\_validate\_filename<!-- {{#callable:fs_validate_filename}} -->
The `fs_validate_filename` function checks if a given filename is valid according to specific criteria, including length, encoding, and character restrictions.
- **Inputs**:
    - `filename`: A constant reference to a `std::string` representing the filename to be validated.
- **Control Flow**:
    - Check if the filename is empty or exceeds 255 characters, returning false if either condition is met.
    - Convert the filename from UTF-8 to UTF-32 using `std::wstring_convert` and check for encoding mismatches, returning false if any are found.
    - Iterate over each character in the UTF-32 encoded filename to check for forbidden codepoints, such as control characters, illegal characters, and specific Unicode characters, returning false if any are found.
    - Check for leading or trailing spaces or a trailing period, returning false if any are present.
    - Check for the presence of ".." or if the filename is ".", returning false if either condition is met.
    - Return true if all checks are passed, indicating the filename is valid.
- **Output**: A boolean value indicating whether the filename is valid (true) or not (false).


---
### fs\_create\_directory\_with\_parents<!-- {{#callable:fs_create_directory_with_parents}} -->
The `fs_create_directory_with_parents` function creates a directory and all its parent directories if they do not already exist, handling both Windows and POSIX systems.
- **Inputs**:
    - `path`: A string representing the path of the directory to be created, including all necessary parent directories.
- **Control Flow**:
    - Check if the code is running on a Windows system using the `_WIN32` preprocessor directive.
    - For Windows, convert the input path from UTF-8 to a wide string using `std::wstring_convert`.
    - Check if the directory already exists using `GetFileAttributesW`; if it exists and is a directory, return true.
    - Iterate through the path, creating each subdirectory using `CreateDirectoryW`.
    - If `CreateDirectoryW` fails, check if the error is `ERROR_ALREADY_EXISTS` and verify the existing path is a directory; otherwise, return false.
    - For non-Windows systems, use `stat` to check if the directory exists and is a directory, returning true if so.
    - Iterate through the path, creating each subdirectory using `mkdir` with permissions `0755`.
    - If `mkdir` fails, return false.
    - Return true if all directories are successfully created or already exist as directories.
- **Output**: A boolean value indicating success (true) if the directory and its parents were created or already exist as directories, or false if an error occurred.


---
### fs\_get\_cache\_directory<!-- {{#callable:fs_get_cache_directory}} -->
The function `fs_get_cache_directory` determines and returns the appropriate cache directory path for the application based on the operating system and environment variables.
- **Inputs**: None
- **Control Flow**:
    - Initialize an empty string `cache_directory`.
    - Define a lambda function `ensure_trailing_slash` to add a trailing slash to a given path if it doesn't already have one.
    - Check if the environment variable `LLAMA_CACHE` is set; if so, set `cache_directory` to its value.
    - If `LLAMA_CACHE` is not set, determine the cache directory based on the operating system:
    - For Linux, FreeBSD, AIX, or OpenBSD, check if `XDG_CACHE_HOME` is set; if so, use its value, otherwise use the `HOME` environment variable with `/.cache/`.
    - For macOS, use the `HOME` environment variable with `/Library/Caches/`.
    - For Windows, use the `LOCALAPPDATA` environment variable.
    - Append `llama.cpp` to the `cache_directory` path.
    - Ensure the final `cache_directory` path has a trailing slash using `ensure_trailing_slash`.
- **Output**: A string representing the cache directory path, with a trailing slash, suitable for the current operating system and environment settings.


---
### fs\_get\_cache\_file<!-- {{#callable:fs_get_cache_file}} -->
The `fs_get_cache_file` function constructs a full path to a cache file by ensuring the cache directory exists and appending the given filename.
- **Inputs**:
    - `filename`: A string representing the name of the file to be cached, which must not contain any directory separators.
- **Control Flow**:
    - Assert that the filename does not contain any directory separators using `GGML_ASSERT`.
    - Retrieve the cache directory path by calling `fs_get_cache_directory()`.
    - Attempt to create the cache directory and its parents by calling `fs_create_directory_with_parents()`.
    - If directory creation fails, throw a `std::runtime_error` with an appropriate error message.
    - Return the full path by concatenating the cache directory path and the filename.
- **Output**: A string representing the full path to the cache file, including the cache directory and the filename.
- **Functions called**:
    - [`fs_get_cache_directory`](#fs_get_cache_directory)
    - [`fs_create_directory_with_parents`](#fs_create_directory_with_parents)


---
### common\_init\_from\_params<!-- {{#callable:common_init_from_params}} -->
The `common_init_from_params` function initializes a model and its context from given parameters, handling various configurations and potential errors during the setup process.
- **Inputs**:
    - `params`: A reference to a `common_params` structure containing configuration settings for model initialization, such as model path, reranking, context shift, control vectors, lora adapters, and sampling options.
- **Control Flow**:
    - Initialize `common_init_result` structure `iparams` to store the result.
    - Convert `params` to model parameters using [`common_model_params_to_llama`](#common_model_params_to_llama).
    - Load the model from file using `llama_model_load_from_file`; log an error and return if loading fails.
    - Retrieve the vocabulary from the loaded model using `llama_model_get_vocab`.
    - If reranking is enabled, check for necessary tokens in the vocabulary and log warnings or errors as needed; free the model and return if checks fail.
    - Convert `params` to context parameters using [`common_context_params_to_llama`](#common_context_params_to_llama).
    - Initialize the context from the model using `llama_init_from_model`; log an error, free the model, and return if initialization fails.
    - Check and adjust context shift settings based on the context's capabilities.
    - If control vectors are specified, load them and apply to the context; free resources and return if loading or application fails.
    - Iterate over lora adapters, initialize and apply each to the model; log errors, free resources, and return if initialization fails.
    - If `lora_init_without_apply` is false, apply lora adapters to the context.
    - Adjust sampling settings based on vocabulary tokens and log warnings if necessary.
    - Set default values for sampling penalties if not specified.
    - If warmup is enabled, perform a warmup run with the model and context.
    - Store the initialized model and context in `iparams` and return `iparams`.
- **Output**: A `common_init_result` structure containing the initialized model and context, or an empty structure if initialization fails.
- **Functions called**:
    - [`common_model_params_to_llama`](#common_model_params_to_llama)
    - [`common_context_params_to_llama`](#common_context_params_to_llama)
    - [`common_control_vector_load`](#common_control_vector_load)
    - [`common_set_adapter_lora`](#common_set_adapter_lora)
    - [`common_token_to_piece`](#common_token_to_piece)


---
### get\_model\_endpoint<!-- {{#callable:get_model_endpoint}} -->
The `get_model_endpoint` function retrieves the model endpoint URL from environment variables or defaults to a predefined URL.
- **Inputs**: None
- **Control Flow**:
    - Retrieve the value of the environment variable 'MODEL_ENDPOINT'.
    - Retrieve the value of the environment variable 'HF_ENDPOINT' for backward compatibility.
    - Determine which environment variable is set, prioritizing 'MODEL_ENDPOINT'.
    - Set the default model endpoint URL to 'https://huggingface.co/'.
    - If an environment variable is set, update the model endpoint URL to its value.
    - Ensure the model endpoint URL ends with a '/' character.
    - Return the final model endpoint URL.
- **Output**: A string representing the model endpoint URL.


---
### common\_set\_adapter\_lora<!-- {{#callable:common_set_adapter_lora}} -->
The `common_set_adapter_lora` function configures LoRA adapters for a given llama context by clearing existing adapters and setting new ones based on a provided list.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure, representing the context in which the LoRA adapters are to be set.
    - `lora`: A reference to a vector of `common_adapter_lora_info` structures, each containing information about a LoRA adapter, including a pointer to the adapter and a scale factor.
- **Control Flow**:
    - Call `llama_clear_adapter_lora` to remove any existing LoRA adapters from the context `ctx`.
    - Iterate over each `common_adapter_lora_info` object in the `lora` vector.
    - For each adapter, check if the `scale` is not zero.
    - If the `scale` is not zero, call `llama_set_adapter_lora` with the context `ctx`, the adapter pointer `la.ptr`, and the scale `la.scale`.
- **Output**: This function does not return a value; it modifies the `llama_context` by setting the specified LoRA adapters.


---
### common\_model\_params\_to\_llama<!-- {{#callable:common_model_params_to_llama}} -->
The function `common_model_params_to_llama` converts a `common_params` structure into a `llama_model_params` structure by mapping and transferring relevant fields.
- **Inputs**:
    - `params`: A reference to a `common_params` structure containing various configuration settings for the model.
- **Control Flow**:
    - Initialize `mparams` with default llama model parameters using `llama_model_default_params()`.
    - Check if `params.devices` is not empty; if so, set `mparams.devices` to point to the data in `params.devices`.
    - If `params.n_gpu_layers` is not -1, set `mparams.n_gpu_layers` to `params.n_gpu_layers`.
    - Directly assign values from `params` to corresponding fields in `mparams` for `main_gpu`, `split_mode`, `tensor_split`, `use_mmap`, `use_mlock`, and `check_tensors`.
    - Check if `params.kv_overrides` is empty; if so, set `mparams.kv_overrides` to NULL, otherwise, ensure the last key is empty and set `mparams.kv_overrides` to the data in `params.kv_overrides`.
    - Check if `params.tensor_buft_overrides` is empty; if so, set `mparams.tensor_buft_overrides` to NULL, otherwise, ensure the last pattern is NULL and set `mparams.tensor_buft_overrides` to the data in `params.tensor_buft_overrides`.
    - Assign `params.load_progress_callback` and `params.load_progress_callback_user_data` to `mparams.progress_callback` and `mparams.progress_callback_user_data`, respectively.
    - Return the populated `mparams` structure.
- **Output**: A `llama_model_params` structure populated with values derived from the input `common_params`.
- **Functions called**:
    - [`llama_model_default_params`](../src/llama-model.cpp.driver.md#llama_model_default_params)


---
### common\_context\_params\_to\_llama<!-- {{#callable:common_context_params_to_llama}} -->
The function `common_context_params_to_llama` converts a `common_params` structure into a `llama_context_params` structure by mapping and assigning corresponding fields.
- **Inputs**:
    - `params`: A `common_params` structure containing various configuration parameters for the context, such as number of contexts, batch sizes, threading information, embedding options, and other model-specific settings.
- **Control Flow**:
    - Initialize a `llama_context_params` structure with default values using `llama_context_default_params()`.
    - Assign values from `params` to corresponding fields in the `cparams` structure, including context size, batch sizes, threading information, and various model-specific parameters.
    - Check if `params.reranking` is true, and if so, set `cparams.embeddings` to true and `cparams.pooling_type` to `LLAMA_POOLING_TYPE_RANK`.
    - Set the cache types `type_k` and `type_v` in `cparams` from `params`.
    - Return the populated `cparams` structure.
- **Output**: A `llama_context_params` structure populated with values from the input `common_params` structure.
- **Functions called**:
    - [`llama_context_default_params`](../src/llama-context.cpp.driver.md#llama_context_default_params)


---
### ggml\_threadpool\_params\_from\_cpu\_params<!-- {{#callable:ggml_threadpool_params_from_cpu_params}} -->
The function `ggml_threadpool_params_from_cpu_params` initializes and returns a `ggml_threadpool_params` structure based on the provided `cpu_params`.
- **Inputs**:
    - `params`: A `cpu_params` structure containing CPU-related parameters such as number of threads, CPU mask, priority, polling, and strict CPU usage.
- **Control Flow**:
    - Initialize a `ggml_threadpool_params` structure named `tpp`.
    - Call [`ggml_threadpool_params_init`](../ggml/src/ggml.c.driver.md#ggml_threadpool_params_init) to set up default values in `tpp` using `params.n_threads`.
    - If `params.mask_valid` is true, copy the CPU mask from `params.cpumask` to `tpp.cpumask` using `std::memcpy`.
    - Set `tpp.prio` to `params.priority`.
    - Set `tpp.poll` to `params.poll`.
    - Set `tpp.strict_cpu` to `params.strict_cpu`.
    - Return the initialized `tpp` structure.
- **Output**: A `ggml_threadpool_params` structure initialized with values from the input `cpu_params`.
- **Functions called**:
    - [`ggml_threadpool_params_init`](../ggml/src/ggml.c.driver.md#ggml_threadpool_params_init)


---
### common\_batch\_clear<!-- {{#callable:common_batch_clear}} -->
The `common_batch_clear` function resets the number of tokens in a `llama_batch` structure to zero.
- **Inputs**:
    - `batch`: A reference to a `llama_batch` structure whose `n_tokens` field is to be reset.
- **Control Flow**:
    - The function takes a reference to a `llama_batch` structure as input.
    - It sets the `n_tokens` field of the `batch` to zero, effectively clearing the batch.
- **Output**: The function does not return any value; it modifies the input `llama_batch` structure in place.


---
### common\_batch\_add<!-- {{#callable:common_batch_add}} -->
The `common_batch_add` function adds a new token, position, sequence IDs, and logits flag to a `llama_batch` structure, incrementing the token count.
- **Inputs**:
    - `batch`: A reference to a `llama_batch` structure where the new token and associated data will be added.
    - `id`: A `llama_token` representing the token ID to be added to the batch.
    - `pos`: A `llama_pos` representing the position of the token within the sequence.
    - `seq_ids`: A constant reference to a vector of `llama_seq_id` representing the sequence IDs associated with the token.
    - `logits`: A boolean flag indicating whether logits are associated with the token.
- **Control Flow**:
    - Assert that the current token count does not exceed the batch size using `GGML_ASSERT`.
    - Assign the token ID `id` to the current position in the `batch.token` array.
    - Assign the position `pos` to the current position in the `batch.pos` array.
    - Set the number of sequence IDs for the current token in `batch.n_seq_id` to the size of `seq_ids`.
    - Iterate over `seq_ids` and assign each sequence ID to the corresponding position in `batch.seq_id`.
    - Assign the `logits` flag to the current position in the `batch.logits` array.
    - Increment the `batch.n_tokens` counter to reflect the addition of the new token.
- **Output**: The function does not return a value; it modifies the `batch` structure in place.


---
### common\_lcp<!-- {{#callable:common_lcp}} -->
The `common_lcp` function calculates the longest common prefix length between two sequences of tokens.
- **Inputs**:
    - `a`: A constant reference to a `llama_tokens` object representing the first sequence of tokens.
    - `b`: A constant reference to a `llama_tokens` object representing the second sequence of tokens.
- **Control Flow**:
    - Initialize a size_t variable `i` to zero.
    - Iterate over the tokens in both sequences `a` and `b` as long as `i` is less than the size of both sequences and the tokens at index `i` in both sequences are equal.
    - Increment `i` in each iteration of the loop.
    - Return the value of `i` after the loop ends.
- **Output**: The function returns a `size_t` value representing the length of the longest common prefix of the two token sequences.


---
### common\_lcs<!-- {{#callable:common_lcs}} -->
The `common_lcs` function calculates the length of the longest common substring between two sequences of tokens.
- **Inputs**:
    - `a`: A constant reference to a `llama_tokens` object representing the first sequence of tokens.
    - `b`: A constant reference to a `llama_tokens` object representing the second sequence of tokens.
- **Control Flow**:
    - Check if either sequence is empty and return 0 if true.
    - Initialize variables for the lengths of the sequences and the maximum LCS length.
    - Create two vectors to store the current and previous row of LCS lengths for space optimization.
    - Iterate over each element in sequence `a` and `b`.
    - If elements at the current positions match, update the current LCS length and possibly the maximum LCS length.
    - If elements do not match, reset the current LCS length to 0.
    - After processing each element in `b`, update the previous row to the current row.
- **Output**: Returns the maximum length of the longest common substring as a `size_t`.


---
### common\_tokenize<!-- {{#callable:common_tokenize}} -->
The `common_tokenize` function tokenizes a given text into a vector of `llama_token` using a specified vocabulary, with options to add or parse special tokens.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that provides the vocabulary for tokenization.
    - `text`: A constant reference to a `std::string` containing the text to be tokenized.
    - `add_special`: A boolean flag indicating whether to add special tokens during tokenization.
    - `parse_special`: A boolean flag indicating whether to parse special tokens during tokenization.
- **Control Flow**:
    - Calculate an upper limit for the number of tokens as the length of the text plus twice the value of `add_special`.
    - Initialize a `std::vector` of `llama_token` with the calculated size.
    - Call `llama_tokenize` to tokenize the text into the `result` vector, updating `n_tokens` with the actual number of tokens.
    - If `n_tokens` is negative, resize the `result` vector to the absolute value of `n_tokens` and re-tokenize, asserting that the re-tokenization result matches the resized size.
    - If `n_tokens` is non-negative, resize the `result` vector to `n_tokens`.
    - Return the `result` vector containing the tokens.
- **Output**: A `std::vector<llama_token>` containing the tokens resulting from the tokenization process.


---
### common\_token\_to\_piece<!-- {{#callable:common_token_to_piece}} -->
The `common_token_to_piece` function converts a given token from a vocabulary into its corresponding string representation, handling special tokens if specified.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure representing the vocabulary from which the token is derived.
    - `token`: A `llama_token` representing the token to be converted into a string piece.
    - `special`: A boolean indicating whether special tokens should be handled differently during conversion.
- **Control Flow**:
    - Initialize an empty string `piece` and resize it to use its internal cache.
    - Call `llama_token_to_piece` to convert the token into a string, storing the result in `piece`.
    - Check if the number of characters `n_chars` returned is negative, indicating the buffer was too small.
    - If `n_chars` is negative, resize `piece` to the required size and call `llama_token_to_piece` again to fill it.
    - Assert that the second call to `llama_token_to_piece` returns the expected negative size, ensuring the buffer was correctly resized.
    - If `n_chars` is non-negative, resize `piece` to the actual number of characters returned.
    - Return the string `piece` containing the token's string representation.
- **Output**: A `std::string` containing the string representation of the given token.


---
### common\_detokenize<!-- {{#callable:common_detokenize}} -->
The `common_detokenize` function converts a sequence of tokens back into a human-readable string using a given vocabulary.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure that contains the vocabulary used for detokenization.
    - `tokens`: A reference to a vector of `llama_token` objects representing the sequence of tokens to be detokenized.
    - `special`: A boolean flag indicating whether special tokens should be considered during detokenization.
- **Control Flow**:
    - Initialize an empty string `text` and resize it to accommodate at least the number of tokens.
    - Call `llama_detokenize` to convert tokens into a string, storing the result in `text`.
    - If the number of characters returned (`n_chars`) is negative, resize `text` to the absolute value of `n_chars` and call `llama_detokenize` again.
    - Assert that the number of characters is within the size of `text` to ensure no overflow occurs.
    - Resize `text` to the actual number of characters (`n_chars`).
    - Return the detokenized string `text`.
- **Output**: A `std::string` containing the detokenized text.


---
### common\_embd\_normalize<!-- {{#callable:common_embd_normalize}} -->
The `common_embd_normalize` function normalizes an input array of floats using a specified norm type and stores the result in an output array.
- **Inputs**:
    - `inp`: A pointer to the input array of floats to be normalized.
    - `out`: A pointer to the output array where the normalized values will be stored.
    - `n`: The number of elements in the input and output arrays.
    - `embd_norm`: An integer specifying the type of normalization to apply: -1 for no normalization, 0 for max absolute normalization, 2 for Euclidean normalization, and any other value for p-norm normalization.
- **Control Flow**:
    - Initialize a double variable `sum` to 0.0.
    - Use a switch statement to determine the normalization type based on `embd_norm`.
    - For `embd_norm` -1, set `sum` to 1.0 (no normalization).
    - For `embd_norm` 0, find the maximum absolute value in the input array and divide it by 32760.0 to set `sum`.
    - For `embd_norm` 2, calculate the Euclidean norm by summing the squares of the input values and taking the square root to set `sum`.
    - For any other `embd_norm`, calculate the p-norm by summing the p-th power of the absolute values of the input and taking the p-th root to set `sum`.
    - Calculate the normalization factor `norm` as 1.0 divided by `sum` if `sum` is greater than 0.0, otherwise set `norm` to 0.0f.
    - Iterate over the input array, multiplying each element by `norm` and storing the result in the output array.
- **Output**: The function does not return a value; it modifies the output array in place with the normalized values.


---
### common\_embd\_similarity\_cos<!-- {{#callable:common_embd_similarity_cos}} -->
The function `common_embd_similarity_cos` calculates the cosine similarity between two embedding vectors.
- **Inputs**:
    - `embd1`: A pointer to the first embedding vector, represented as an array of floats.
    - `embd2`: A pointer to the second embedding vector, represented as an array of floats.
    - `n`: The number of elements in each embedding vector.
- **Control Flow**:
    - Initialize three double variables `sum`, `sum1`, and `sum2` to 0.0.
    - Iterate over each element of the embedding vectors from 0 to n-1.
    - For each element, update `sum` with the product of the corresponding elements from `embd1` and `embd2`.
    - Update `sum1` with the square of the current element from `embd1` and `sum2` with the square of the current element from `embd2`.
    - Check if either `sum1` or `sum2` is zero, indicating one or both vectors are zero vectors.
    - If both `sum1` and `sum2` are zero, return 1.0f as two zero vectors are considered similar.
    - If only one of them is zero, return 0.0f as the vectors are not similar.
    - If neither is zero, return the cosine similarity calculated as `sum / (sqrt(sum1) * sqrt(sum2))`.
- **Output**: A float representing the cosine similarity between the two embedding vectors, ranging from 0.0 to 1.0.


---
### common\_control\_vector\_load\_one<!-- {{#callable:common_control_vector_load_one}} -->
The function `common_control_vector_load_one` loads a control vector from a file and processes its tensors to produce a `common_control_vector_data` structure.
- **Inputs**:
    - `load_info`: A `common_control_vector_load_info` structure containing the filename and strength for loading the control vector.
- **Control Flow**:
    - Initialize a `common_control_vector_data` result with default values.
    - Create a `gguf_context` from the file specified in `load_info` using [`gguf_init_from_file`](../ggml/src/gguf.cpp.driver.md#gguf_init_from_file).
    - If the context is not created, log an error and return the default result.
    - Retrieve the number of tensors in the context using [`gguf_get_n_tensors`](../ggml/src/gguf.cpp.driver.md#gguf_get_n_tensors).
    - Iterate over each tensor, extracting its name and attempting to parse a layer index from it.
    - If the layer index is invalid or zero, log an error and break the loop.
    - Retrieve the tensor using [`ggml_get_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_tensor) and check its type and dimensions.
    - If the tensor type is not `GGML_TYPE_F32` or its dimensions are not 1D, log an error and break the loop.
    - If the result's `n_embd` is uninitialized, set it to the number of elements in the tensor; otherwise, ensure it matches the current tensor's element count.
    - Resize the result's data vector if necessary and accumulate the tensor's data into the result's data, scaled by `load_info.strength`.
    - If the result's `n_embd` is still uninitialized after processing, log a warning and clear the result's data.
    - Free the `gguf_context` and `ggml_context` resources.
    - Return the populated `common_control_vector_data` result.
- **Output**: A `common_control_vector_data` structure containing the loaded and processed control vector data, or default values if loading failed.
- **Functions called**:
    - [`gguf_init_from_file`](../ggml/src/gguf.cpp.driver.md#gguf_init_from_file)
    - [`gguf_get_n_tensors`](../ggml/src/gguf.cpp.driver.md#gguf_get_n_tensors)
    - [`gguf_get_tensor_name`](../ggml/src/gguf.cpp.driver.md#gguf_get_tensor_name)
    - [`ggml_get_tensor`](../ggml/src/ggml.c.driver.md#ggml_get_tensor)
    - [`ggml_n_dims`](../ggml/src/ggml.c.driver.md#ggml_n_dims)
    - [`ggml_nelements`](../ggml/src/ggml.c.driver.md#ggml_nelements)
    - [`gguf_free`](../ggml/src/gguf.cpp.driver.md#gguf_free)
    - [`ggml_free`](../ggml/src/ggml.c.driver.md#ggml_free)


---
### common\_control\_vector\_load<!-- {{#callable:common_control_vector_load}} -->
The function `common_control_vector_load` loads and aggregates control vector data from multiple files, ensuring consistency in dimensions and handling errors if any occur.
- **Inputs**:
    - `load_infos`: A constant reference to a vector of `common_control_vector_load_info` structures, each containing information about a control vector file to be loaded.
- **Control Flow**:
    - Initialize `result` with `n_embd` set to -1 and an empty data vector.
    - Iterate over each `info` in `load_infos`.
    - For each `info`, call [`common_control_vector_load_one`](#common_control_vector_load_one) to load the control vector data.
    - If the loaded control vector (`cur`) has `n_embd` of -1, set `result.n_embd` to -1 and break the loop.
    - If `result.n_embd` is not -1 and does not match `cur.n_embd`, log an error, set `result.n_embd` to -1, and break the loop.
    - If `result.n_embd` is -1, move `cur` into `result`.
    - Otherwise, resize `result.data` to accommodate the new data if necessary, and add `cur.data` to `result.data`.
    - After the loop, if `result.n_embd` is -1, log an error and clear `result.data`.
- **Output**: Returns a `common_control_vector_data` structure containing the aggregated control vector data and its embedding dimension (`n_embd`).
- **Functions called**:
    - [`common_control_vector_load_one`](#common_control_vector_load_one)


---
### common\_opt\_dataset\_init<!-- {{#callable:common_opt_dataset_init}} -->
The `common_opt_dataset_init` function initializes an optimization dataset for a given context and token sequence with a specified stride.
- **Inputs**:
    - `ctx`: A pointer to a `llama_context` structure, representing the context for which the dataset is being initialized.
    - `tokens`: A constant reference to a vector of `llama_token` objects, representing the sequence of tokens to be used in the dataset.
    - `stride`: An integer value representing the stride length used to determine the spacing between data points in the token sequence.
- **Control Flow**:
    - Calculate `ne_datapoint` as the number of context elements using `llama_n_ctx(ctx)`.
    - Calculate `ndata` as the number of data points by dividing the size of `tokens` minus `ne_datapoint` minus 1 by `stride`.
    - Initialize a `ggml_opt_dataset_t` object `result` with the calculated `ne_datapoint` and `ndata`.
    - Obtain pointers to the data and labels arrays from the `result` dataset.
    - Iterate over each data point index `idata` from 0 to `ndata - 1`.
    - For each `idata`, copy `ne_datapoint` tokens from the `tokens` vector to the `data` array, starting at `idata * stride`.
    - For each `idata`, copy `ne_datapoint` tokens from the `tokens` vector to the `labels` array, starting at `idata * stride + 1`.
    - Return the initialized `ggml_opt_dataset_t` object `result`.
- **Output**: A `ggml_opt_dataset_t` object representing the initialized optimization dataset with data and labels populated from the token sequence.
- **Functions called**:
    - [`ggml_opt_dataset_data`](../ggml/src/ggml-opt.cpp.driver.md#ggml_opt_dataset_data)
    - [`ggml_opt_dataset_labels`](../ggml/src/ggml-opt.cpp.driver.md#ggml_opt_dataset_labels)


