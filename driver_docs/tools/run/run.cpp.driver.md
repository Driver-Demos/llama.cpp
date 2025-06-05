# Purpose
This C++ source code file is designed to facilitate the execution of a language model, specifically a large language model (LLM) referred to as "llama." The file includes functionality for initializing and configuring the model, handling user input, and generating responses based on prompts. It is structured to be an executable program, as indicated by the presence of a [`main`](#main) function, which serves as the entry point for execution. The code integrates various components such as command-line argument parsing, file handling, and network operations, which are essential for loading and running the model.

The code is organized around several key classes and functions. The `Opt` class is responsible for parsing and storing command-line options, which configure the model's behavior, such as context size, number of threads, and verbosity. The `LlamaData` class manages the model, context, and sampler, and includes methods for initializing these components. The file also includes platform-specific code for handling signals and terminal interactions, ensuring compatibility across different operating systems. Additionally, the code supports downloading model files from various sources, including HTTP, GitHub, and S3, using the `HttpClient` class when compiled with CURL support. The program is designed to be interactive, allowing users to input prompts and receive generated responses, with support for chat templates to format the interaction.
# Imports and Dependencies

---
- `chat.h`
- `common.h`
- `llama-cpp.h`
- `log.h`
- `linenoise.cpp/linenoise.h`
- `nlohmann/json.hpp`
- `windows.h`
- `io.h`
- `sys/file.h`
- `sys/ioctl.h`
- `unistd.h`
- `curl/curl.h`
- `signal.h`
- `climits`
- `cstdarg`
- `cstdio`
- `cstring`
- `filesystem`
- `iostream`
- `list`
- `sstream`
- `string`
- `vector`


# Data Structures

---
### Opt<!-- {{#data_structure:Opt}} -->
- **Type**: `class`
- **Members**:
    - `ctx_params`: Holds the context parameters for the llama model.
    - `model_params`: Stores the model parameters for the llama model.
    - `model_`: Represents the model identifier or path.
    - `chat_template_file`: Specifies the path to the chat template file.
    - `user`: Stores the user identifier or name.
    - `use_jinja`: Indicates whether Jinja templating is used for the chat template.
    - `context_size`: Defines the context size for the model, defaulting to -1.
    - `ngl`: Specifies the number of GPU layers, defaulting to -1.
    - `n_threads`: Indicates the number of threads to use, defaulting to -1.
    - `temperature`: Sets the temperature for sampling, defaulting to -1.
    - `verbose`: Determines if verbose logging is enabled.
- **Description**: The `Opt` class is designed to manage and initialize options and parameters for running a llama model. It includes attributes for context and model parameters, user and model identifiers, and various configuration settings such as context size, number of GPU layers, threads, and temperature. The class also handles command-line argument parsing to set these options, and provides functionality to display help information. It is integral to setting up the environment and parameters needed for executing the llama model operations.
- **Member Functions**:
    - [`Opt::init`](#Optinit)
    - [`Opt::parse_flag`](#Optparse_flag)
    - [`Opt::handle_option_with_value`](#Opthandle_option_with_value)
    - [`Opt::handle_option_with_value`](#Opthandle_option_with_value)
    - [`Opt::handle_option_with_value`](#Opthandle_option_with_value)
    - [`Opt::parse_options_with_value`](#Optparse_options_with_value)
    - [`Opt::parse_options`](#Optparse_options)
    - [`Opt::parse_positional_args`](#Optparse_positional_args)
    - [`Opt::parse`](#Optparse)
    - [`Opt::print_help`](#Optprint_help)

**Methods**

---
#### Opt::init<!-- {{#callable:Opt::init}} -->
The `init` function initializes the `Opt` class by setting default parameters, parsing command-line arguments, and configuring context and model parameters based on the input arguments.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize default context and model parameters using [`llama_context_default_params`](../../src/llama-context.cpp.driver.md#llama_context_default_params) and [`llama_model_default_params`](../../src/llama-model.cpp.driver.md#llama_model_default_params) functions.
    - Set default values for context size, number of threads, number of GPU layers, and temperature from the initialized parameters.
    - Check if the number of arguments (`argc`) is less than 2, print an error message, display help, and return 1 if true.
    - Call the [`parse`](#Optparse) function to process command-line arguments; if parsing fails, print an error message, display help, and return 1.
    - Check if the help flag is set, display help, and return 2 if true.
    - Set context and model parameters based on parsed values or default values if the parsed values are negative.
    - Return 0 to indicate successful initialization.
- **Output**: Returns an integer indicating the success or failure of the initialization process: 0 for success, 1 for failure due to argument issues, and 2 if help is requested.
- **Functions called**:
    - [`llama_context_default_params`](../../src/llama-context.cpp.driver.md#llama_context_default_params)
    - [`llama_model_default_params`](../../src/llama-model.cpp.driver.md#llama_model_default_params)
    - [`printe`](#printe)
    - [`Opt::print_help`](#Optprint_help)
    - [`Opt::parse`](#Optparse)
- **See also**: [`Opt`](#Opt)  (Data Structure)


---
#### Opt::parse\_flag<!-- {{#callable:Opt::parse_flag}} -->
The `parse_flag` function checks if a given command-line argument matches either a short or long option flag.
- **Inputs**:
    - `argv`: A pointer to an array of C-style strings representing command-line arguments.
    - `i`: An integer index indicating the current position in the `argv` array to check.
    - `short_opt`: A C-style string representing the short option flag to compare against.
    - `long_opt`: A C-style string representing the long option flag to compare against.
- **Control Flow**:
    - The function uses `strcmp` to compare the argument at index `i` in `argv` with `short_opt` and `long_opt`.
    - It returns `true` if either comparison is equal (i.e., the argument matches one of the option flags), otherwise it returns `false`.
- **Output**: A boolean value indicating whether the argument at the specified index matches either the short or long option flag.
- **See also**: [`Opt`](#Opt)  (Data Structure)


---
#### Opt::handle\_option\_with\_value<!-- {{#callable:Opt::handle_option_with_value}} -->
The function `handle_option_with_value` processes a command-line option that requires a value, updating the option value and advancing the argument index.
- **Inputs**:
    - `argc`: The total number of command-line arguments.
    - `argv`: An array of C-style strings representing the command-line arguments.
    - `i`: A reference to the current index in the argument list, which will be incremented if a value is successfully processed.
    - `option_value`: A reference to an integer where the parsed option value will be stored.
- **Control Flow**:
    - Check if the next argument index (i + 1) is out of bounds (greater than or equal to argc).
    - If out of bounds, return 1 to indicate an error.
    - Otherwise, increment the index `i` and convert the next argument to an integer using `std::atoi`, storing it in `option_value`.
    - Return 0 to indicate success.
- **Output**: Returns 0 on success, indicating the option value was successfully parsed, or 1 if there was an error due to insufficient arguments.
- **See also**: [`Opt`](#Opt)  (Data Structure)


---
#### Opt::handle\_option\_with\_value<!-- {{#callable:Opt::handle_option_with_value}} -->
The function `handle_option_with_value` processes a command-line argument that requires a float value and updates the provided reference with this value.
- **Inputs**:
    - `argc`: The total number of command-line arguments.
    - `argv`: An array of C-style strings representing the command-line arguments.
    - `i`: A reference to the current index in the argument list, which will be incremented if a value is successfully processed.
    - `option_value`: A reference to a float variable where the parsed float value from the command-line argument will be stored.
- **Control Flow**:
    - Check if the next argument index (i + 1) is within the bounds of the argument count (argc).
    - If the next argument index is out of bounds, return 1 to indicate an error.
    - Increment the index `i` and convert the next argument to a float using `std::atof`, storing the result in `option_value`.
    - Return 0 to indicate successful processing of the option value.
- **Output**: Returns 0 if the option value is successfully processed, or 1 if there is an error (e.g., missing value).
- **See also**: [`Opt`](#Opt)  (Data Structure)


---
#### Opt::handle\_option\_with\_value<!-- {{#callable:Opt::handle_option_with_value}} -->
The function `handle_option_with_value` retrieves the next command-line argument as a string and assigns it to the provided reference variable.
- **Inputs**:
    - `argc`: The total number of command-line arguments.
    - `argv`: An array of C-style strings representing the command-line arguments.
    - `i`: A reference to the current index in the `argv` array, which will be incremented to point to the next argument.
    - `option_value`: A reference to a string where the next argument's value will be stored.
- **Control Flow**:
    - Check if the current index `i` plus one is greater than or equal to `argc`, indicating there are no more arguments to process.
    - If there are no more arguments, return 1 to indicate an error or missing value.
    - Increment the index `i` to point to the next argument.
    - Assign the next argument from `argv` to `option_value`.
    - Return 0 to indicate successful retrieval of the option value.
- **Output**: Returns 0 if the next argument is successfully retrieved and assigned to `option_value`, otherwise returns 1 if there are no more arguments to process.
- **See also**: [`Opt`](#Opt)  (Data Structure)


---
#### Opt::parse\_options\_with\_value<!-- {{#callable:Opt::parse_options_with_value}} -->
The `parse_options_with_value` function processes command-line options that require a value, updating the corresponding variables and returning a status code based on the success of the operation.
- **Inputs**:
    - `argc`: The number of command-line arguments.
    - `argv`: An array of C-style strings representing the command-line arguments.
    - `i`: A reference to the current index in the `argv` array, which is updated as options are parsed.
    - `options_parsing`: A boolean reference indicating whether options parsing is currently active.
- **Control Flow**:
    - Check if options parsing is active and the current argument matches '-c' or '--context-size'.
    - If matched, call [`handle_option_with_value`](#Opthandle_option_with_value) to parse the value for `context_size`; return 1 if it fails.
    - Check if options parsing is active and the current argument matches '-n', '-ngl', or '--ngl'.
    - If matched, call [`handle_option_with_value`](#Opthandle_option_with_value) to parse the value for `ngl`; return 1 if it fails.
    - Check if options parsing is active and the current argument matches '-t' or '--threads'.
    - If matched, call [`handle_option_with_value`](#Opthandle_option_with_value) to parse the value for `n_threads`; return 1 if it fails.
    - Check if options parsing is active and the current argument matches '--temp'.
    - If matched, call [`handle_option_with_value`](#Opthandle_option_with_value) to parse the value for `temperature`; return 1 if it fails.
    - Check if options parsing is active and the current argument matches '--chat-template-file'.
    - If matched, call [`handle_option_with_value`](#Opthandle_option_with_value) to parse the value for `chat_template_file`, set `use_jinja` to true; return 1 if it fails.
    - If none of the conditions are met, return 2 indicating an unrecognized option.
- **Output**: Returns 0 on successful parsing of an option with a value, 1 if parsing fails due to a missing value, or 2 if the option is unrecognized.
- **Functions called**:
    - [`Opt::handle_option_with_value`](#Opthandle_option_with_value)
- **See also**: [`Opt`](#Opt)  (Data Structure)


---
#### Opt::parse\_options<!-- {{#callable:Opt::parse_options}} -->
The `parse_options` function processes command-line options from an argument list, setting flags and returning status codes based on the options encountered.
- **Inputs**:
    - `argv`: A pointer to an array of C-style strings representing command-line arguments.
    - `i`: A reference to an integer representing the current index in the argument list.
    - `options_parsing`: A reference to a boolean indicating whether options parsing is currently active.
- **Control Flow**:
    - Check if options parsing is active and if the current argument matches '-v', '--verbose', or '--log-verbose', then set the verbose flag to true.
    - Check if options parsing is active and if the current argument is '--jinja', then set the use_jinja flag to true.
    - Check if options parsing is active and if the current argument matches '-h' or '--help', then set the help flag to true and return 0.
    - Check if options parsing is active and if the current argument is '--', then set options_parsing to false.
    - If none of the above conditions are met, return 2 indicating an unrecognized option.
- **Output**: Returns 0 if an option is successfully parsed or if help is requested, and 2 if an unrecognized option is encountered.
- **Functions called**:
    - [`Opt::parse_flag`](#Optparse_flag)
- **See also**: [`Opt`](#Opt)  (Data Structure)


---
#### Opt::parse\_positional\_args<!-- {{#callable:Opt::parse_positional_args}} -->
The `parse_positional_args` function processes positional command-line arguments to set the model and user variables based on their order of appearance.
- **Inputs**:
    - `argv`: A constant character pointer array representing the command-line arguments.
    - `i`: A reference to an integer representing the current index in the `argv` array.
    - `positional_args_i`: A reference to an integer tracking the number of positional arguments processed so far.
- **Control Flow**:
    - Check if `positional_args_i` is 0, indicating the first positional argument is expected.
    - If the first character of `argv[i]` is null or '-', return 1 indicating an error or end of positional arguments.
    - Increment `positional_args_i` and set `model_` to `argv[i]` if it's the first positional argument.
    - If `positional_args_i` is 1, increment it and set `user` to `argv[i]`.
    - For subsequent positional arguments, append `argv[i]` to `user` with a space separator.
    - Return 0 to indicate successful parsing of the current argument.
- **Output**: Returns an integer, 0 for successful parsing or 1 if an error or end of positional arguments is encountered.
- **See also**: [`Opt`](#Opt)  (Data Structure)


---
#### Opt::parse<!-- {{#callable:Opt::parse}} -->
The `parse` function processes command-line arguments, parsing options and positional arguments, and returns a status code indicating success or failure.
- **Inputs**:
    - `argc`: The number of command-line arguments.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize a boolean `options_parsing` to true to track if options are still being parsed.
    - Iterate over the command-line arguments starting from index 1.
    - For each argument, attempt to parse it as an option with a value using [`parse_options_with_value`](#Optparse_options_with_value).
    - If [`parse_options_with_value`](#Optparse_options_with_value) returns 0, continue to the next argument; if it returns 1, return 1 indicating an error.
    - If the argument is not an option with a value, attempt to parse it as a simple option using [`parse_options`](#Optparse_options).
    - If [`parse_options`](#Optparse_options) returns 0, continue to the next argument; if it returns 1, return 1 indicating an error.
    - If the argument is neither an option with a value nor a simple option, attempt to parse it as a positional argument using [`parse_positional_args`](#Optparse_positional_args).
    - If [`parse_positional_args`](#Optparse_positional_args) returns 1, return 1 indicating an error.
    - After processing all arguments, check if the `model_` string is empty; if so, return 1 indicating an error.
    - Return 0 to indicate successful parsing.
- **Output**: An integer status code: 0 for successful parsing, 1 for an error in parsing.
- **Functions called**:
    - [`Opt::parse_options_with_value`](#Optparse_options_with_value)
    - [`Opt::parse_options`](#Optparse_options)
    - [`Opt::parse_positional_args`](#Optparse_positional_args)
- **See also**: [`Opt`](#Opt)  (Data Structure)


---
#### Opt::print\_help<!-- {{#callable:Opt::print_help}} -->
The `print_help` function displays a help message detailing the usage, options, commands, and examples for running a language model using the `llama-run` command.
- **Inputs**: None
- **Control Flow**:
    - The function uses `printf` to output a formatted help message to the console.
    - The message includes sections for description, usage, options, commands, and examples.
    - Default values for context size, number of GPU layers, temperature, and number of threads are inserted into the message using format specifiers.
- **Output**: The function does not return any value; it outputs the help message directly to the console.
- **See also**: [`Opt`](#Opt)  (Data Structure)



---
### progress\_data<!-- {{#data_structure:progress_data}} -->
- **Type**: `struct`
- **Members**:
    - `file_size`: Represents the size of the file being processed, initialized to 0.
    - `start_time`: Stores the time point when the progress tracking started, initialized to the current time.
    - `printed`: Indicates whether the progress has been printed, initialized to false.
- **Description**: The `progress_data` struct is designed to track the progress of a file-related operation, such as downloading or processing a file. It contains information about the file size, the start time of the operation, and whether the progress has been printed to the console. This struct is useful for monitoring and displaying the progress of long-running file operations in a user interface or log.


---
### File<!-- {{#data_structure:File}} -->
- **Type**: `class`
- **Members**:
    - `file`: A pointer to a FILE object, initialized to nullptr.
    - `fd`: An integer file descriptor, initialized to -1.
    - `hFile`: A HANDLE object for Windows file operations, initialized to nullptr.
- **Description**: The `File` class is a utility for handling file operations, including opening, locking, and reading files. It encapsulates a file pointer (`FILE * file`) and provides methods to open a file with a specified mode, lock the file for exclusive access, and convert the file's contents to a string. The class also manages file descriptors (`fd`) and, on Windows systems, a handle (`hFile`) for file locking operations. The destructor ensures that any open file is properly closed and any locks are released.

**Methods**

---
#### File::open<!-- {{#callable:File::open}} -->
The `open` function opens a file with the specified filename and mode using the [`ggml_fopen`](../../ggml/src/ggml.c.driver.md#ggml_fopen) function and returns a pointer to the opened file.
- **Inputs**:
    - `filename`: A constant reference to a `std::string` representing the name of the file to be opened.
    - `mode`: A constant pointer to a `char` representing the mode in which the file should be opened (e.g., "r" for read, "w" for write).
- **Control Flow**:
    - The function calls [`ggml_fopen`](../../ggml/src/ggml.c.driver.md#ggml_fopen) with the `filename` converted to a C-style string and the `mode` to open the file.
    - The result of [`ggml_fopen`](../../ggml/src/ggml.c.driver.md#ggml_fopen) is assigned to the `file` member variable of the `File` class.
    - The function returns the `file` pointer.
- **Output**: A pointer to a `FILE` object representing the opened file, or `nullptr` if the file could not be opened.
- **Functions called**:
    - [`ggml_fopen`](../../ggml/src/ggml.c.driver.md#ggml_fopen)
- **See also**: [`File`](linenoise.cpp/linenoise.cpp.driver.md#File)  (Data Structure)


---
#### File::lock<!-- {{#callable:File::lock}} -->
The `lock` function attempts to acquire an exclusive lock on a file associated with the `File` class instance, using platform-specific mechanisms.
- **Inputs**: None
- **Control Flow**:
    - Check if the `file` member is not null, indicating an open file.
    - On Windows, retrieve the file descriptor and handle, and attempt to lock the file using `LockFileEx`; if unsuccessful, set `fd` to -1 and return 1.
    - On non-Windows systems, retrieve the file descriptor and attempt to lock the file using `flock`; if unsuccessful, set `fd` to -1 and return 1.
    - If the lock is successful, return 0.
- **Output**: Returns 0 if the lock is successfully acquired, otherwise returns 1 if the lock attempt fails.
- **See also**: [`File`](linenoise.cpp/linenoise.cpp.driver.md#File)  (Data Structure)


---
#### File::to\_string<!-- {{#callable:File::to_string}} -->
The `to_string` method reads the entire content of a file and returns it as a string.
- **Inputs**: None
- **Control Flow**:
    - Seek to the end of the file to determine its size using `fseek` and `ftell`.
    - Reset the file position to the beginning using `fseek`.
    - Resize a string to hold the file's content based on the determined size.
    - Read the file's content into the string using `fread`.
    - Check if the read size matches the file size; if not, print an error message.
- **Output**: A `std::string` containing the entire content of the file.
- **Functions called**:
    - [`printe`](#printe)
- **See also**: [`File`](linenoise.cpp/linenoise.cpp.driver.md#File)  (Data Structure)


---
#### File::\~File<!-- {{#callable:File::~File}} -->
The destructor `~File()` releases file locks and closes the file if it is open.
- **Inputs**: None
- **Control Flow**:
    - Check if the file descriptor `fd` is valid (greater than or equal to 0).
    - If on Windows (`_WIN32`), check if `hFile` is not `INVALID_HANDLE_VALUE` and unlock the file using `UnlockFileEx`.
    - If not on Windows, unlock the file using `flock` with `LOCK_UN`.
    - Check if the `file` pointer is not null, and if so, close the file using `fclose`.
- **Output**: The function does not return any value as it is a destructor.
- **See also**: [`File`](linenoise.cpp/linenoise.cpp.driver.md#File)  (Data Structure)



---
### HttpClient<!-- {{#data_structure:HttpClient}} -->
- **Type**: `class`
- **Members**:
    - `curl`: A pointer to a CURL object used for handling HTTP requests.
    - `chunk`: A pointer to a curl_slist structure used for managing HTTP headers.
- **Description**: The `HttpClient` class is designed to facilitate HTTP requests using the libcurl library. It provides functionality to initialize a connection, set various options such as headers and progress tracking, and perform the HTTP request. The class manages resources like CURL handles and header lists, ensuring they are properly cleaned up in the destructor. It supports downloading data to a file or capturing it into a string, with options for resuming downloads and displaying progress.
- **Member Functions**:
    - [`HttpClient::init`](#HttpClientinit)
    - [`HttpClient::~HttpClient`](#HttpClientHttpClient)
    - [`HttpClient::set_write_options`](#HttpClientset_write_options)
    - [`HttpClient::set_resume_point`](#HttpClientset_resume_point)
    - [`HttpClient::set_progress_options`](#HttpClientset_progress_options)
    - [`HttpClient::set_headers`](#HttpClientset_headers)
    - [`HttpClient::perform`](#HttpClientperform)
    - [`HttpClient::human_readable_time`](#HttpClienthuman_readable_time)
    - [`HttpClient::human_readable_size`](#HttpClienthuman_readable_size)
    - [`HttpClient::update_progress`](#HttpClientupdate_progress)
    - [`HttpClient::calculate_percentage`](#HttpClientcalculate_percentage)
    - [`HttpClient::generate_progress_prefix`](#HttpClientgenerate_progress_prefix)
    - [`HttpClient::calculate_speed`](#HttpClientcalculate_speed)
    - [`HttpClient::generate_progress_suffix`](#HttpClientgenerate_progress_suffix)
    - [`HttpClient::calculate_progress_bar_width`](#HttpClientcalculate_progress_bar_width)
    - [`HttpClient::generate_progress_bar`](#HttpClientgenerate_progress_bar)
    - [`HttpClient::print_progress`](#HttpClientprint_progress)
    - [`HttpClient::write_data`](#HttpClientwrite_data)
    - [`HttpClient::capture_data`](#HttpClientcapture_data)

**Methods**

---
#### HttpClient::init<!-- {{#callable:HttpClient::init}} -->
The `init` function initializes an HTTP client to download a resource from a given URL, handling file writing, progress tracking, and response capturing.
- **Inputs**:
    - `url`: A string representing the URL of the resource to be downloaded.
    - `headers`: A vector of strings containing HTTP headers to be included in the request.
    - `output_file`: A string representing the path to the file where the downloaded content will be saved.
    - `progress`: A boolean indicating whether to display download progress.
    - `response_str`: An optional pointer to a string where the response data will be stored if not writing to a file.
- **Control Flow**:
    - Check if the output file already exists; if so, return 0 to indicate no action is needed.
    - Initialize a CURL handle; if initialization fails, return 1 to indicate an error.
    - If an output file is specified, create a partial file for writing and lock it exclusively; if any step fails, return 1.
    - Set options for writing data either to a file or to a response string based on the presence of `response_str`.
    - Determine the resume point for downloading based on the existing partial file size.
    - Configure progress tracking if `progress` is true.
    - Set the HTTP headers for the request.
    - Perform the HTTP request using CURL; if it fails, print an error message and return 1.
    - If the download is successful and an output file is specified, rename the partial file to the final output file name.
    - Return 0 to indicate successful completion.
- **Output**: An integer indicating the success (0) or failure (1) of the initialization and download process.
- **Functions called**:
    - [`printe`](#printe)
    - [`HttpClient::set_write_options`](#HttpClientset_write_options)
    - [`HttpClient::set_resume_point`](#HttpClientset_resume_point)
    - [`HttpClient::set_progress_options`](#HttpClientset_progress_options)
    - [`HttpClient::set_headers`](#HttpClientset_headers)
    - [`HttpClient::perform`](#HttpClientperform)
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::\~HttpClient<!-- {{#callable:HttpClient::~HttpClient}} -->
The destructor `~HttpClient` cleans up resources by freeing the curl slist and cleaning up the CURL handle if they are not null.
- **Inputs**: None
- **Control Flow**:
    - Check if `chunk` is not null; if true, free all elements in the curl slist using `curl_slist_free_all(chunk)`.
    - Check if `curl` is not null; if true, clean up the CURL handle using `curl_easy_cleanup(curl)`.
- **Output**: This destructor does not return any value; it performs cleanup operations to release resources.
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::set\_write\_options<!-- {{#callable:HttpClient::set_write_options}} -->
The `set_write_options` function configures the CURL write options to either capture data into a string or write data to a file based on the provided arguments.
- **Inputs**:
    - `response_str`: A pointer to a `std::string` where the response data will be captured if not null.
    - `out`: A constant reference to a `File` object representing the file where data will be written if `response_str` is null.
- **Control Flow**:
    - Check if `response_str` is not null.
    - If `response_str` is not null, set the CURL write function to `capture_data` and the write data target to `response_str`.
    - If `response_str` is null, set the CURL write function to `write_data` and the write data target to `out.file`.
- **Output**: The function does not return a value; it sets CURL options for writing data.
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::set\_resume\_point<!-- {{#callable:HttpClient::set_resume_point}} -->
The `set_resume_point` function determines the size of an existing file and sets the resume point for a download operation using libcurl.
- **Inputs**:
    - `output_file`: A constant reference to a string representing the path to the output file.
- **Control Flow**:
    - Initialize `file_size` to 0.
    - Check if the file specified by `output_file` exists using `std::filesystem::exists`.
    - If the file exists, get its size using `std::filesystem::file_size` and assign it to `file_size`.
    - Set the resume point for the curl operation using `curl_easy_setopt` with `CURLOPT_RESUME_FROM_LARGE` and the file size cast to `curl_off_t`.
- **Output**: Returns the size of the file as a `size_t`.
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::set\_progress\_options<!-- {{#callable:HttpClient::set_progress_options}} -->
The `set_progress_options` function configures the CURL handle to display progress information during a file transfer if the `progress` flag is set to true.
- **Inputs**:
    - `progress`: A boolean flag indicating whether progress information should be displayed during the file transfer.
    - `data`: A reference to a `progress_data` structure that holds information related to the progress of the file transfer.
- **Control Flow**:
    - Check if the `progress` flag is true.
    - If true, set the CURL option `CURLOPT_NOPROGRESS` to 0L to enable progress information.
    - Set the CURL option `CURLOPT_XFERINFODATA` to point to the `data` structure.
    - Set the CURL option `CURLOPT_XFERINFOFUNCTION` to use the `update_progress` function for progress updates.
- **Output**: This function does not return any value.
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::set\_headers<!-- {{#callable:HttpClient::set_headers}} -->
The `set_headers` function configures HTTP headers for a CURL request by appending each header from a given list to a CURL slist and setting it as the HTTPHEADER option for a CURL handle.
- **Inputs**:
    - `headers`: A constant reference to a vector of strings, where each string represents an HTTP header to be set for the CURL request.
- **Control Flow**:
    - Check if the `headers` vector is not empty.
    - If `chunk` is not null, free the existing list of headers using `curl_slist_free_all` and set `chunk` to null.
    - Iterate over each header in the `headers` vector, appending it to the `chunk` list using `curl_slist_append`.
    - Set the `chunk` list as the HTTPHEADER option for the CURL handle using `curl_easy_setopt`.
- **Output**: The function does not return a value; it modifies the `chunk` member variable and the CURL handle's HTTPHEADER option.
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::perform<!-- {{#callable:HttpClient::perform}} -->
The `perform` function configures and executes a cURL request to a specified URL with specific options set for following redirects, using HTTPS as the default protocol, and failing on HTTP errors.
- **Inputs**:
    - `url`: A constant reference to a `std::string` representing the URL to which the cURL request will be made.
- **Control Flow**:
    - Set the cURL option `CURLOPT_URL` to the provided URL string converted to a C-style string using `c_str()`.
    - Enable following of HTTP redirects by setting `CURLOPT_FOLLOWLOCATION` to 1L.
    - Set the default protocol to HTTPS by setting `CURLOPT_DEFAULT_PROTOCOL` to "https".
    - Configure the cURL request to fail on HTTP errors by setting `CURLOPT_FAILONERROR` to 1L.
    - Execute the cURL request using `curl_easy_perform` and return its result.
- **Output**: Returns a `CURLcode` which indicates the result of the cURL operation, with `CURLE_OK` indicating success and other codes indicating various errors.
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::human\_readable\_time<!-- {{#callable:HttpClient::human_readable_time}} -->
The `human_readable_time` function converts a time duration in seconds into a human-readable string format of hours, minutes, and seconds.
- **Inputs**:
    - `seconds`: A double representing the time duration in seconds to be converted into a human-readable format.
- **Control Flow**:
    - Convert the input `seconds` to an integer and calculate the number of hours by dividing by 3600.
    - Calculate the remaining minutes by taking the modulus of `seconds` with 3600 and then dividing by 60.
    - Calculate the remaining seconds by taking the modulus of `seconds` with 60.
    - If the number of hours is greater than zero, format the time as "Xh Ym Zs".
    - If the number of hours is zero but minutes are greater than zero, format the time as "Ym Zs".
    - If both hours and minutes are zero, format the time as "Zs".
- **Output**: A string representing the time in a human-readable format, such as "1h 02m 03s", "02m 03s", or "03s".
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::human\_readable\_size<!-- {{#callable:HttpClient::human_readable_size}} -->
The `human_readable_size` function converts a file size in bytes to a human-readable string format with appropriate size suffixes (B, KB, MB, GB, TB).
- **Inputs**:
    - `size`: A `curl_off_t` type representing the size in bytes to be converted to a human-readable format.
- **Control Flow**:
    - Initialize an array of suffixes representing byte units (B, KB, MB, GB, TB).
    - Calculate the number of suffixes available.
    - Initialize a double variable `dbl_size` with the input size.
    - Check if the size is greater than 1024 bytes.
    - If true, iterate through the suffixes, dividing the size by 1024 each time, until the size is less than 1024 or the last suffix is reached.
    - Format the final size and suffix into a string with two decimal places.
- **Output**: A `std::string` representing the size in a human-readable format with the appropriate suffix.
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::update\_progress<!-- {{#callable:HttpClient::update_progress}} -->
The `update_progress` function updates and displays the download progress of a file using various metrics such as percentage completed, download speed, and estimated time remaining.
- **Inputs**:
    - `ptr`: A pointer to a `progress_data` structure that holds information about the download progress.
    - `total_to_download`: The total size of the file to be downloaded, in bytes.
    - `now_downloaded`: The current amount of data downloaded, in bytes.
    - `curl_off_t`: Unused parameter.
    - `curl_off_t`: Unused parameter.
- **Control Flow**:
    - Cast the `ptr` to a `progress_data` pointer to access download progress data.
    - Check if `total_to_download` is less than or equal to zero; if so, return 0 immediately.
    - Add the file size from `progress_data` to `total_to_download` to account for any previously downloaded data.
    - Calculate the total downloaded amount including the file size and compute the download percentage.
    - Generate a progress prefix string based on the calculated percentage.
    - Calculate the download speed using the current downloaded amount and the start time from `progress_data`.
    - Estimate the remaining time for the download based on the speed and remaining data.
    - Generate a progress suffix string that includes the current download size, total size, speed, and estimated time.
    - Calculate the width of the progress bar based on the terminal width and the lengths of the prefix and suffix.
    - Generate the progress bar string based on the calculated width and percentage.
    - Print the progress using the prefix, progress bar, and suffix.
    - Set the `printed` flag in `progress_data` to true to indicate that progress has been printed.
    - Return 0 to indicate successful execution.
- **Output**: The function returns an integer value of 0, indicating successful execution.
- **Functions called**:
    - [`HttpClient::calculate_percentage`](#HttpClientcalculate_percentage)
    - [`HttpClient::generate_progress_prefix`](#HttpClientgenerate_progress_prefix)
    - [`HttpClient::calculate_speed`](#HttpClientcalculate_speed)
    - [`HttpClient::generate_progress_suffix`](#HttpClientgenerate_progress_suffix)
    - [`HttpClient::calculate_progress_bar_width`](#HttpClientcalculate_progress_bar_width)
    - [`HttpClient::generate_progress_bar`](#HttpClientgenerate_progress_bar)
    - [`HttpClient::print_progress`](#HttpClientprint_progress)
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::calculate\_percentage<!-- {{#callable:HttpClient::calculate_percentage}} -->
The `calculate_percentage` function computes the percentage of a given amount downloaded relative to the total amount to be downloaded.
- **Inputs**:
    - `now_downloaded_plus_file_size`: The current amount downloaded plus any previously downloaded file size, represented as a `curl_off_t` type.
    - `total_to_download`: The total amount to be downloaded, represented as a `curl_off_t` type.
- **Control Flow**:
    - The function multiplies `now_downloaded_plus_file_size` by 100 to convert the ratio to a percentage.
    - It then divides the result by `total_to_download` to get the percentage of the download completed.
- **Output**: The function returns a `curl_off_t` value representing the percentage of the download completed.
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::generate\_progress\_prefix<!-- {{#callable:HttpClient::generate_progress_prefix}} -->
The `generate_progress_prefix` function formats a given percentage value into a string representation suitable for a progress display.
- **Inputs**:
    - `percentage`: A `curl_off_t` type representing the percentage of progress to be formatted.
- **Control Flow**:
    - The function takes a `percentage` as input.
    - It casts the `percentage` to a `long int`.
    - It uses `string_format` to format the percentage into a string with a specific format: a three-digit percentage followed by a percentage sign and a pipe character, e.g., ' 50% |'.
    - The formatted string is returned.
- **Output**: A `std::string` representing the formatted progress percentage.
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::calculate\_speed<!-- {{#callable:HttpClient::calculate_speed}} -->
The `calculate_speed` function computes the download speed based on the amount of data downloaded and the elapsed time since the start of the download.
- **Inputs**:
    - `now_downloaded`: The amount of data downloaded so far, measured in bytes, represented as a `curl_off_t` type.
    - `start_time`: The time point when the download started, represented as a `std::chrono::steady_clock::time_point`.
- **Control Flow**:
    - Capture the current time using `std::chrono::steady_clock::now()` and store it in `now`.
    - Calculate the elapsed time in seconds by subtracting `start_time` from `now` and converting the duration to a double.
    - Return the download speed by dividing `now_downloaded` by the elapsed time in seconds.
- **Output**: The function returns a `double` representing the download speed in bytes per second.
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::generate\_progress\_suffix<!-- {{#callable:HttpClient::generate_progress_suffix}} -->
The `generate_progress_suffix` function formats and returns a string representing the current download progress, total download size, download speed, and estimated time remaining in a human-readable format.
- **Inputs**:
    - `now_downloaded_plus_file_size`: The current amount of data downloaded plus the size of any existing file, represented as a `curl_off_t` type.
    - `total_to_download`: The total size of the data to be downloaded, represented as a `curl_off_t` type.
    - `speed`: The current download speed, represented as a `double`.
    - `estimated_time`: The estimated time remaining for the download to complete, represented as a `double`.
- **Control Flow**:
    - The function defines a constant integer `width` set to 10, which is used for formatting the output string.
    - It calls [`human_readable_size`](#HttpClienthuman_readable_size) to convert `now_downloaded_plus_file_size`, `total_to_download`, and `speed` into human-readable strings.
    - It calls [`human_readable_time`](#HttpClienthuman_readable_time) to convert `estimated_time` into a human-readable string.
    - It uses `string_format` to format these values into a single string with specified widths, separated by slashes and ending with '/s' for speed and time.
- **Output**: A formatted string representing the download progress, total size, speed, and estimated time in a human-readable format.
- **Functions called**:
    - [`HttpClient::human_readable_size`](#HttpClienthuman_readable_size)
    - [`HttpClient::human_readable_time`](#HttpClienthuman_readable_time)
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::calculate\_progress\_bar\_width<!-- {{#callable:HttpClient::calculate_progress_bar_width}} -->
The `calculate_progress_bar_width` function computes the width of a progress bar based on the terminal width and the sizes of the progress prefix and suffix strings.
- **Inputs**:
    - `progress_prefix`: A string representing the prefix of the progress bar, which typically includes the percentage completed.
    - `progress_suffix`: A string representing the suffix of the progress bar, which typically includes additional information like download speed or estimated time remaining.
- **Control Flow**:
    - Calculate the initial progress bar width by subtracting the sizes of the prefix, suffix, and additional fixed characters (3) from the terminal width.
    - Check if the calculated progress bar width is less than 1; if so, set it to 1 to ensure a minimum width.
    - Return the calculated or adjusted progress bar width.
- **Output**: An integer representing the width of the progress bar, ensuring it is at least 1.
- **Functions called**:
    - [`get_terminal_width`](#get_terminal_width)
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::generate\_progress\_bar<!-- {{#callable:HttpClient::generate_progress_bar}} -->
The `generate_progress_bar` function creates a visual representation of progress as a string of filled and empty blocks based on a given percentage and width.
- **Inputs**:
    - `progress_bar_width`: An integer representing the total width of the progress bar in characters.
    - `percentage`: A `curl_off_t` type representing the percentage of completion, ranging from 0 to 100.
    - `progress_bar`: A reference to a `std::string` where the generated progress bar will be appended.
- **Control Flow**:
    - Calculate the position `pos` in the progress bar where the filled blocks should end, based on the given percentage and width.
    - Iterate over the range from 0 to `progress_bar_width`, appending a filled block ('█') to `progress_bar` if the current index is less than `pos`, otherwise append a space (' ').
- **Output**: Returns the `progress_bar` string with the appended progress bar representation.
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::print\_progress<!-- {{#callable:HttpClient::print_progress}} -->
The `print_progress` function outputs a formatted progress line to the standard error stream, combining a prefix, a progress bar, and a suffix.
- **Inputs**:
    - `progress_prefix`: A string representing the prefix of the progress line, typically indicating the percentage completed.
    - `progress_bar`: A string representing the visual progress bar, usually composed of characters like '█' to indicate progress.
    - `progress_suffix`: A string representing the suffix of the progress line, typically containing additional information like speed or estimated time remaining.
- **Control Flow**:
    - The function uses the [`printe`](#printe) function to print to the standard error stream.
    - It uses a carriage return (`\r`) to overwrite the current line in the terminal.
    - The `LOG_CLR_TO_EOL` macro is used to clear the line to the end, ensuring no residual characters remain from previous outputs.
    - The function formats the output string by concatenating the `progress_prefix`, `progress_bar`, and `progress_suffix` with a separator '|'.
- **Output**: The function does not return any value; it directly outputs the formatted progress line to the standard error stream.
- **Functions called**:
    - [`printe`](#printe)
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::write\_data<!-- {{#callable:HttpClient::write_data}} -->
The `write_data` function writes data from a memory buffer to a file stream using the `fwrite` function.
- **Inputs**:
    - `ptr`: A pointer to the data buffer that needs to be written to the file.
    - `size`: The size of each element to be written, in bytes.
    - `nmemb`: The number of elements, each of size `size`, to be written.
    - `stream`: A pointer to a `FILE` object that identifies the output stream where the data is to be written.
- **Control Flow**:
    - The function casts the `stream` pointer to a `FILE*` type.
    - It then calls the `fwrite` function to write `nmemb` elements of `size` bytes each from the buffer pointed to by `ptr` to the file stream `out`.
- **Output**: The function returns the total number of elements successfully written, which is less than `nmemb` only if a write error occurs.
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)


---
#### HttpClient::capture\_data<!-- {{#callable:HttpClient::capture_data}} -->
The `capture_data` function appends data from a memory buffer to a `std::string` object.
- **Inputs**:
    - `ptr`: A pointer to the memory buffer containing the data to be appended.
    - `size`: The size of each data element in the buffer.
    - `nmemb`: The number of data elements in the buffer.
    - `stream`: A pointer to a `std::string` object where the data will be appended.
- **Control Flow**:
    - Cast the `stream` pointer to a `std::string` pointer.
    - Append the data from the `ptr` buffer to the `std::string` object, using the total size calculated as `size * nmemb`.
    - Return the total number of bytes appended, which is `size * nmemb`.
- **Output**: The function returns the total number of bytes appended to the `std::string`, which is `size * nmemb`.
- **See also**: [`HttpClient`](#HttpClient)  (Data Structure)



---
### LlamaData<!-- {{#data_structure:LlamaData}} -->
- **Type**: `class`
- **Members**:
    - `model`: A pointer to the llama model used for processing.
    - `sampler`: A pointer to the llama sampler used for sampling operations.
    - `context`: A pointer to the llama context used for managing the model's context.
    - `messages`: A vector of llama chat messages, possibly to be switched to a common chat message format.
    - `msg_strs`: A list of strings representing message contents.
    - `fmtted`: A vector of characters used for formatted data.
- **Description**: The `LlamaData` class is designed to encapsulate the components necessary for managing and interacting with a llama model, including the model itself, a sampler, and a context. It also maintains a collection of chat messages and their string representations, as well as a buffer for formatted data. The class provides an initialization function to set up these components using specified options, facilitating the use of the llama model in chat applications.
- **Member Functions**:
    - [`LlamaData::init`](#LlamaDatainit)
    - [`LlamaData::download`](#LlamaDatadownload)
    - [`LlamaData::download`](#LlamaDatadownload)
    - [`LlamaData::extract_model_and_tag`](#LlamaDataextract_model_and_tag)
    - [`LlamaData::download_and_parse_manifest`](#LlamaDatadownload_and_parse_manifest)
    - [`LlamaData::dl_from_endpoint`](#LlamaDatadl_from_endpoint)
    - [`LlamaData::modelscope_dl`](#LlamaDatamodelscope_dl)
    - [`LlamaData::huggingface_dl`](#LlamaDatahuggingface_dl)
    - [`LlamaData::ollama_dl`](#LlamaDataollama_dl)
    - [`LlamaData::github_dl`](#LlamaDatagithub_dl)
    - [`LlamaData::s3_dl`](#LlamaDatas3_dl)
    - [`LlamaData::basename`](#LlamaDatabasename)
    - [`LlamaData::rm_until_substring`](#LlamaDatarm_until_substring)
    - [`LlamaData::resolve_model`](#LlamaDataresolve_model)
    - [`LlamaData::initialize_model`](#LlamaDatainitialize_model)
    - [`LlamaData::initialize_context`](#LlamaDatainitialize_context)
    - [`LlamaData::initialize_sampler`](#LlamaDatainitialize_sampler)

**Methods**

---
#### LlamaData::init<!-- {{#callable:LlamaData::init}} -->
The `init` function initializes the model, context, and sampler for a `LlamaData` object using the provided options.
- **Inputs**:
    - `opt`: An `Opt` object containing configuration parameters for initializing the model, context, and sampler.
- **Control Flow**:
    - Call [`initialize_model`](#LlamaDatainitialize_model) with `opt` to initialize the model and assign it to `model`.
    - Check if `model` is null; if so, return 1 indicating failure.
    - Call [`initialize_context`](#LlamaDatainitialize_context) with `model` and `opt` to initialize the context and assign it to `context`.
    - Check if `context` is null; if so, return 1 indicating failure.
    - Call [`initialize_sampler`](#LlamaDatainitialize_sampler) with `opt` to initialize the sampler and assign it to `sampler`.
    - Return 0 indicating successful initialization.
- **Output**: Returns an integer: 0 on successful initialization, or 1 if either the model or context initialization fails.
- **Functions called**:
    - [`LlamaData::initialize_model`](#LlamaDatainitialize_model)
    - [`LlamaData::initialize_context`](#LlamaDatainitialize_context)
    - [`LlamaData::initialize_sampler`](#LlamaDatainitialize_sampler)
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::download<!-- {{#callable:LlamaData::download}} -->
The `download` function attempts to download a file from a specified URL using HTTP, with optional progress display and custom headers, and writes the response to a file or a string.
- **Inputs**:
    - `url`: A string representing the URL from which to download the file.
    - `output_file`: A string representing the path to the file where the downloaded content should be saved.
    - `progress`: A boolean indicating whether to display download progress.
    - `headers`: An optional vector of strings representing HTTP headers to include in the request.
    - `response_str`: An optional pointer to a string where the response content will be stored if not writing to a file.
- **Control Flow**:
    - An `HttpClient` object is instantiated.
    - The `init` method of `HttpClient` is called with the provided URL, headers, output file, progress flag, and response string pointer.
    - If the `init` method returns a non-zero value, indicating an error, the function returns 1.
    - If the `init` method succeeds, the function returns 0.
- **Output**: An integer indicating success (0) or failure (1) of the download operation.
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::download<!-- {{#callable:LlamaData::download}} -->
The `download` function attempts to download a file from a given URL but returns an error message if the program is built without libcurl support.
- **Inputs**:
    - `url`: A string representing the URL from which to download the file.
    - `output_file`: A string representing the path where the downloaded file should be saved.
    - `progress`: A boolean indicating whether to show download progress.
    - `headers`: An optional vector of strings representing HTTP headers to include in the request.
    - `response_str`: An optional pointer to a string where the response data will be stored.
- **Control Flow**:
    - The function checks if the program is built with libcurl support using the `#ifdef LLAMA_USE_CURL` directive.
    - If libcurl is not supported, it prints an error message indicating that downloading is not supported and returns 1.
- **Output**: The function returns an integer, 1, indicating that the download operation is not supported without libcurl.
- **Functions called**:
    - [`printe`](#printe)
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::extract\_model\_and\_tag<!-- {{#callable:LlamaData::extract_model_and_tag}} -->
The `extract_model_and_tag` function extracts the model name and tag from a given model string and constructs a URL using a base URL.
- **Inputs**:
    - `model`: A reference to a string containing the model name and optionally a tag separated by a colon.
    - `base_url`: A constant reference to a string representing the base URL to be used for constructing the final URL.
- **Control Flow**:
    - Initialize `model_tag` to "latest".
    - Find the position of the colon in the `model` string.
    - If a colon is found, extract the substring after the colon as `model_tag` and update `model` to the substring before the colon.
    - Construct the URL by concatenating `base_url`, `model`, "/manifests/", and `model_tag`.
    - Return a pair containing the updated `model` and the constructed URL.
- **Output**: A `std::pair` containing the extracted model name and the constructed URL.
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::download\_and\_parse\_manifest<!-- {{#callable:LlamaData::download_and_parse_manifest}} -->
The `download_and_parse_manifest` function downloads a JSON manifest from a given URL and parses it into a JSON object.
- **Inputs**:
    - `url`: A string representing the URL from which the manifest will be downloaded.
    - `headers`: A vector of strings representing HTTP headers to be included in the download request.
    - `manifest`: A reference to a nlohmann::json object where the parsed manifest will be stored.
- **Control Flow**:
    - Initialize an empty string `manifest_str` to store the downloaded manifest content.
    - Call the [`download`](#LlamaDatadownload) function with the provided URL, headers, and a pointer to `manifest_str` to download the manifest content.
    - Check the return value of the [`download`](#LlamaDatadownload) function; if it is non-zero, return this value as an error code.
    - Parse the downloaded manifest string into the `manifest` JSON object using `nlohmann::json::parse`.
    - Return 0 to indicate success.
- **Output**: Returns an integer, 0 on success or a non-zero error code if the download fails.
- **Functions called**:
    - [`LlamaData::download`](#LlamaDatadownload)
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::dl\_from\_endpoint<!-- {{#callable:LlamaData::dl_from_endpoint}} -->
The `dl_from_endpoint` function downloads a model file from a specified endpoint, handling URL construction and manifest parsing if necessary.
- **Inputs**:
    - `model_endpoint`: A reference to a string representing the base URL of the model endpoint.
    - `model`: A reference to a string representing the model identifier, which may include a path or tag.
    - `bn`: A constant string representing the base name for the output file.
- **Control Flow**:
    - The function first attempts to find the second occurrence of '/' in the `model` string to determine if it contains a path.
    - If no second '/' is found, it uses [`extract_model_and_tag`](#LlamaDataextract_model_and_tag) to get the model name and manifest URL, then downloads and parses the manifest to get the file name.
    - If a second '/' is found, it splits the `model` into `hfr` (head) and `hff` (file name) based on the position of the '/' character.
    - Constructs the full URL by appending `hfr` and `hff` to the `model_endpoint` with a specific path structure.
    - Calls the [`download`](#LlamaDatadownload) function with the constructed URL, output file name `bn`, and headers to perform the download.
- **Output**: Returns an integer status code, where 0 indicates success and non-zero indicates an error during the download process.
- **Functions called**:
    - [`LlamaData::extract_model_and_tag`](#LlamaDataextract_model_and_tag)
    - [`LlamaData::download_and_parse_manifest`](#LlamaDatadownload_and_parse_manifest)
    - [`LlamaData::download`](#LlamaDatadownload)
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::modelscope\_dl<!-- {{#callable:LlamaData::modelscope_dl}} -->
The `modelscope_dl` function downloads a model from the ModelScope endpoint using a specified model name and base name for the output file.
- **Inputs**:
    - `model`: A reference to a string representing the model name, which may include a tag separated by a colon.
    - `bn`: A constant string representing the base name for the output file where the model will be saved.
- **Control Flow**:
    - Initialize the `model_endpoint` string with the URL 'https://modelscope.cn/models/'.
    - Call the [`dl_from_endpoint`](#LlamaDatadl_from_endpoint) function with `model_endpoint`, `model`, and `bn` as arguments.
    - Return the result of the [`dl_from_endpoint`](#LlamaDatadl_from_endpoint) function call.
- **Output**: Returns an integer status code from the [`dl_from_endpoint`](#LlamaDatadl_from_endpoint) function, indicating success or failure of the download operation.
- **Functions called**:
    - [`LlamaData::dl_from_endpoint`](#LlamaDatadl_from_endpoint)
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::huggingface\_dl<!-- {{#callable:LlamaData::huggingface_dl}} -->
The `huggingface_dl` function downloads a model from a Hugging Face endpoint using a specified model name and base name.
- **Inputs**:
    - `model`: A reference to a string representing the model name, which may include a tag separated by a colon.
    - `bn`: A constant reference to a string representing the base name for the downloaded file.
- **Control Flow**:
    - Retrieve the model endpoint URL by calling `get_model_endpoint()`.
    - Call the [`dl_from_endpoint`](#LlamaDatadl_from_endpoint) function with the model endpoint, model name, and base name to perform the download.
- **Output**: Returns an integer status code from the [`dl_from_endpoint`](#LlamaDatadl_from_endpoint) function, indicating success or failure of the download operation.
- **Functions called**:
    - [`LlamaData::dl_from_endpoint`](#LlamaDatadl_from_endpoint)
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::ollama\_dl<!-- {{#callable:LlamaData::ollama_dl}} -->
The `ollama_dl` function downloads a model from the Ollama registry by constructing the appropriate URLs for the model's manifest and blob, and then downloading the blob using HTTP requests.
- **Inputs**:
    - `model`: A reference to a string representing the model name, which may be modified to include a default 'library/' prefix if it lacks a '/' character.
    - `bn`: A constant reference to a string representing the base name for the output file where the downloaded model will be saved.
- **Control Flow**:
    - Initialize HTTP headers to accept Docker manifest v2 JSON format.
    - Check if the model string contains a '/', and if not, prepend 'library/' to the model name.
    - Extract the model name and manifest URL using the [`extract_model_and_tag`](#LlamaDataextract_model_and_tag) function with the base URL 'https://registry.ollama.ai/v2/'.
    - Download and parse the manifest JSON from the manifest URL using [`download_and_parse_manifest`](#LlamaDatadownload_and_parse_manifest).
    - If the download or parsing fails, return the error code.
    - Iterate over the layers in the manifest to find the layer with media type 'application/vnd.ollama.image.model' and extract its digest.
    - Construct the blob URL using the model name and the extracted layer digest.
    - Download the blob from the constructed blob URL using the [`download`](#LlamaDatadownload) function with the specified headers and return the result.
- **Output**: Returns an integer indicating success (0) or failure (non-zero error code) of the download operation.
- **Functions called**:
    - [`LlamaData::extract_model_and_tag`](#LlamaDataextract_model_and_tag)
    - [`LlamaData::download_and_parse_manifest`](#LlamaDatadownload_and_parse_manifest)
    - [`LlamaData::download`](#LlamaDatadownload)
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::github\_dl<!-- {{#callable:LlamaData::github_dl}} -->
The `github_dl` function downloads a file from a specified GitHub repository and branch, constructing the URL based on the provided model string and saving the file with a given base name.
- **Inputs**:
    - `model`: A string representing the GitHub repository and optionally the branch, formatted as 'repository@branch'.
    - `bn`: A string representing the base name for the downloaded file.
- **Control Flow**:
    - Initialize the repository and branch variables from the model string, defaulting the branch to 'main'.
    - Check if the model string contains an '@' character to separate the repository and branch, and update the variables accordingly.
    - Split the repository string by '/' to extract the organization and project names.
    - Validate that the repository string is in a valid format with at least three parts; if not, print an error and return 1.
    - Construct the URL for the raw file download from GitHub using the organization, project, and branch information.
    - Call the [`download`](#LlamaDatadownload) function with the constructed URL, base name, and a progress flag set to true, returning its result.
- **Output**: Returns 0 on successful download, or 1 if there is an error in the repository format or during the download process.
- **Functions called**:
    - [`printe`](#printe)
    - [`LlamaData::download`](#LlamaDatadownload)
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::s3\_dl<!-- {{#callable:LlamaData::s3_dl}} -->
The `s3_dl` function downloads a file from an Amazon S3 bucket using AWS Signature Version 4 for authentication.
- **Inputs**:
    - `model`: A string representing the S3 path in the format 'bucket/key'.
    - `bn`: A string representing the base name for the output file.
- **Control Flow**:
    - Check if the 'model' string contains a '/' to separate bucket and key; return 1 if not found.
    - Extract the bucket name and key from the 'model' string.
    - Retrieve AWS credentials from environment variables; return 1 if not found.
    - Generate AWS Signature Version 4 headers using the current date and time.
    - Construct the S3 URL using the bucket and key.
    - Call the 'download' function with the constructed URL, output file name, and headers.
- **Output**: Returns 0 on successful download, or 1 if an error occurs during the process.
- **Functions called**:
    - [`printe`](#printe)
    - [`strftime_fmt`](#strftime_fmt)
    - [`LlamaData::download`](#LlamaDatadownload)
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::basename<!-- {{#callable:LlamaData::basename}} -->
The `basename` function extracts the file name from a given file path by removing the directory components.
- **Inputs**:
    - `path`: A string representing the file path from which the base name (file name) is to be extracted.
- **Control Flow**:
    - Find the last occurrence of either '/' or '\' in the input path string to determine the position of the last directory separator.
    - Check if no directory separator is found; if so, return the entire path as it is already a base name.
    - If a directory separator is found, return the substring starting just after the last separator, which represents the base name.
- **Output**: A string representing the base name (file name) extracted from the input path.
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::rm\_until\_substring<!-- {{#callable:LlamaData::rm_until_substring}} -->
The `rm_until_substring` function modifies a given string by removing all characters up to and including a specified substring, if the substring is found.
- **Inputs**:
    - `model_`: A reference to a string that will be modified by removing characters up to and including the specified substring.
    - `substring`: A constant reference to the substring that the function will search for within the string `model_`.
- **Control Flow**:
    - Find the position of the first occurrence of `substring` in `model_` using `find` method.
    - Check if the position is `npos`, indicating that the substring was not found; if so, return 1.
    - If the substring is found, modify `model_` to be the substring starting immediately after the found substring.
    - Return 0 to indicate successful modification.
- **Output**: An integer value: 0 if the substring was found and the string was modified, or 1 if the substring was not found.
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::resolve\_model<!-- {{#callable:LlamaData::resolve_model}} -->
The `resolve_model` function processes a model URL or path, determines its source, and downloads it if necessary, returning a status code.
- **Inputs**:
    - `model_`: A reference to a string representing the model's URL or file path, which will be modified in place.
- **Control Flow**:
    - Initialize a return value `ret` to 0.
    - Check if the model path starts with 'file://' or if the file exists in the filesystem; if so, remove the '://' prefix and return `ret`.
    - Extract the base name of the model path into `bn`.
    - Check if the model path starts with 'hf://', 'huggingface://', or 'hf.co/'; if so, remove the 'hf.co/' and '://' prefixes, download the model using [`huggingface_dl`](#LlamaDatahuggingface_dl), and update `ret`.
    - Check if the model path starts with 'ms://' or 'modelscope://'; if so, remove the '://' prefix, download the model using [`modelscope_dl`](#LlamaDatamodelscope_dl), and update `ret`.
    - Check if the model path starts with 'https://' or 'http://' and is not from 'https://ollama.com/library/'; if so, download the model using [`download`](#LlamaDatadownload) and update `ret`.
    - Check if the model path starts with 'github:' or 'github://'; if so, remove the 'github:' and '://' prefixes, download the model using [`github_dl`](#LlamaDatagithub_dl), and update `ret`.
    - Check if the model path starts with 's3://'; if so, remove the '://' prefix, download the model using [`s3_dl`](#LlamaDatas3_dl), and update `ret`.
    - For any other case (assumed to be 'ollama://' or no prefix), remove the 'ollama.com/library/' and '://' prefixes, download the model using [`ollama_dl`](#LlamaDataollama_dl), and update `ret`.
    - Set `model_` to the base name `bn`.
    - Return the status code `ret`.
- **Output**: An integer status code indicating success (0) or failure (non-zero) of the model resolution and download process.
- **Functions called**:
    - [`LlamaData::rm_until_substring`](#LlamaDatarm_until_substring)
    - [`LlamaData::basename`](#LlamaDatabasename)
    - [`LlamaData::huggingface_dl`](#LlamaDatahuggingface_dl)
    - [`LlamaData::modelscope_dl`](#LlamaDatamodelscope_dl)
    - [`LlamaData::download`](#LlamaDatadownload)
    - [`LlamaData::github_dl`](#LlamaDatagithub_dl)
    - [`LlamaData::s3_dl`](#LlamaDatas3_dl)
    - [`LlamaData::ollama_dl`](#LlamaDataollama_dl)
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::initialize\_model<!-- {{#callable:LlamaData::initialize_model}} -->
The `initialize_model` function initializes and loads a model from a file based on the provided options.
- **Inputs**:
    - `opt`: An instance of the `Opt` class containing model parameters and the model file path.
- **Control Flow**:
    - Call `ggml_backend_load_all()` to load all necessary backends.
    - Call `resolve_model(opt.model_)` to resolve the model path or download it if necessary.
    - Print a message indicating the model is being loaded.
    - Attempt to load the model from the file using `llama_model_load_from_file` with the resolved model path and parameters.
    - Check if the model was successfully loaded; if not, print an error message.
    - Clear the loading message from the console.
    - Return the loaded model.
- **Output**: Returns a `llama_model_ptr`, which is a unique pointer to the loaded model, or `nullptr` if the model could not be loaded.
- **Functions called**:
    - [`ggml_backend_load_all`](../../ggml/src/ggml-backend-reg.cpp.driver.md#ggml_backend_load_all)
    - [`LlamaData::resolve_model`](#LlamaDataresolve_model)
    - [`printe`](#printe)
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::initialize\_context<!-- {{#callable:LlamaData::initialize_context}} -->
The `initialize_context` function initializes a llama context using a given model and options, and returns a pointer to the created context.
- **Inputs**:
    - `model`: A constant reference to a `llama_model_ptr`, which is a smart pointer to the llama model used for context initialization.
    - `opt`: A constant reference to an `Opt` object, which contains various options and parameters, including context parameters (`ctx_params`) used for initializing the context.
- **Control Flow**:
    - Create a `llama_context_ptr` named `context` by calling `llama_init_from_model` with the model and context parameters from `opt`.
    - Check if `context` is null, indicating a failure to create the context.
    - If `context` is null, print an error message indicating the failure to create the llama context.
    - Return the `context` pointer, which may be null if initialization failed.
- **Output**: A `llama_context_ptr`, which is a smart pointer to the initialized llama context, or null if initialization failed.
- **Functions called**:
    - [`printe`](#printe)
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)


---
#### LlamaData::initialize\_sampler<!-- {{#callable:LlamaData::initialize_sampler}} -->
The `initialize_sampler` function initializes and configures a sampler for the Llama model using specified parameters.
- **Inputs**:
    - `opt`: An instance of the `Opt` class containing configuration options, specifically the temperature setting for the sampler.
- **Control Flow**:
    - Create a `llama_sampler_ptr` object `sampler` initialized with default parameters using `llama_sampler_chain_init` and `llama_sampler_chain_default_params`.
    - Add a minimum probability sampler to the sampler chain with parameters 0.05f and 1 using `llama_sampler_chain_add` and `llama_sampler_init_min_p`.
    - Add a temperature-based sampler to the sampler chain using the temperature from `opt` with `llama_sampler_chain_add` and `llama_sampler_init_temp`.
    - Add a distribution-based sampler to the sampler chain using a default seed with `llama_sampler_chain_add` and `llama_sampler_init_dist`.
    - Return the configured `sampler` object.
- **Output**: A `llama_sampler_ptr` object representing the initialized and configured sampler.
- **See also**: [`LlamaData`](#LlamaData)  (Data Structure)



# Functions

---
### sigint\_handler<!-- {{#callable:sigint_handler}} -->
The `sigint_handler` function handles the SIGINT signal by printing a newline and exiting the program.
- **Inputs**:
    - `int`: The signal number, typically SIGINT, which is passed to the handler when the signal is caught.
- **Control Flow**:
    - The function is marked with [[noreturn]], indicating it will not return to the caller.
    - It prints a newline character followed by a default log color reset.
    - The function then calls `exit(0)` to terminate the program immediately.
- **Output**: The function does not return any value as it is marked [[noreturn]] and exits the program.


---
### printe<!-- {{#callable:printe}} -->
The `printe` function is a variadic function that formats and prints a message to the standard error stream.
- **Inputs**:
    - `fmt`: A C-style string that contains the format string for the message to be printed.
    - `...`: A variable number of arguments that are used to replace format specifiers in the format string.
- **Control Flow**:
    - Initialize a `va_list` variable `args` to handle the variable arguments.
    - Start processing the variable arguments using `va_start`, with `fmt` as the last fixed argument.
    - Use `vfprintf` to print the formatted message to `stderr`, passing `fmt` and `args` as arguments.
    - End processing of the variable arguments using `va_end`.
    - Return the result of `vfprintf`, which is the number of characters printed.
- **Output**: The function returns an integer representing the number of characters printed to `stderr`.


---
### strftime\_fmt<!-- {{#callable:strftime_fmt}} -->
The `strftime_fmt` function formats a `std::tm` time structure into a string based on a given format string.
- **Inputs**:
    - `fmt`: A C-style string representing the format to be used for formatting the time.
    - `tm`: A `std::tm` structure representing the time to be formatted.
- **Control Flow**:
    - Create a `std::ostringstream` object to build the formatted string.
    - Use `std::put_time` to format the `std::tm` object `tm` according to the format string `fmt` and insert it into the `ostringstream`.
    - Convert the contents of the `ostringstream` to a `std::string` and return it.
- **Output**: A `std::string` containing the formatted time.


---
### get\_terminal\_width<!-- {{#callable:get_terminal_width}} -->
The `get_terminal_width` function retrieves the current width of the terminal window in characters.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the code is being compiled for a Windows environment using the `_WIN32` macro.
    - If on Windows, it uses `GetConsoleScreenBufferInfo` to obtain the console screen buffer information and calculates the width by subtracting the left position from the right position of the window and adding one.
    - If not on Windows, it uses the `ioctl` function with `TIOCGWINSZ` to get the window size and returns the number of columns (`ws_col`).
- **Output**: The function returns an integer representing the width of the terminal window in characters.


---
### add\_message<!-- {{#callable:add_message}} -->
The `add_message` function appends a message with a specified role and text to the `LlamaData` structure's message list.
- **Inputs**:
    - `role`: A constant character pointer representing the role of the message sender, such as 'user' or 'assistant'.
    - `text`: A constant reference to a string containing the message text to be added.
    - `llama_data`: A reference to a `LlamaData` object where the message will be stored.
- **Control Flow**:
    - The function begins by moving the `text` string into the `msg_strs` list of the `llama_data` object.
    - It then accesses the last element of `msg_strs` to get the C-style string representation of the text.
    - Finally, it appends a new `llama_chat_message` to the `messages` vector of `llama_data`, using the `role` and the C-style string of the text.
- **Output**: The function does not return any value; it modifies the `llama_data` object by adding a new message.


---
### apply\_chat\_template<!-- {{#callable:apply_chat_template}} -->
The `apply_chat_template` function applies a chat template to a set of messages in `LlamaData`, optionally appending a generation prompt and using Jinja templating, and returns the size of the resulting formatted prompt.
- **Inputs**:
    - `tmpls`: A pointer to a `common_chat_templates` structure containing the chat templates to be applied.
    - `llama_data`: A reference to a `LlamaData` object that contains the messages to which the chat template will be applied.
    - `append`: A boolean flag indicating whether to append a generation prompt to the messages.
    - `use_jinja`: A boolean flag indicating whether to use Jinja templating when applying the chat template.
- **Control Flow**:
    - Initialize a `common_chat_templates_inputs` object named `inputs`.
    - Iterate over each message in `llama_data.messages`, convert it to a `common_chat_msg`, and add it to `inputs.messages`.
    - Set `inputs.add_generation_prompt` to the value of `append` and `inputs.use_jinja` to the value of `use_jinja`.
    - Call `common_chat_templates_apply` with `tmpls` and `inputs` to get `chat_params`.
    - Extract the `prompt` from `chat_params` and store it in `result`.
    - Resize `llama_data.fmtted` to accommodate the size of `result` plus one for the null terminator.
    - Copy the contents of `result` into `llama_data.fmtted`.
    - Return the size of `result`.
- **Output**: The function returns an integer representing the size of the formatted prompt after applying the chat template.


---
### tokenize\_prompt<!-- {{#callable:tokenize_prompt}} -->
The `tokenize_prompt` function tokenizes a given prompt string into a vector of llama tokens using a specified vocabulary and context data.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure, which contains the vocabulary used for tokenization.
    - `prompt`: A constant reference to a `std::string` representing the prompt text to be tokenized.
    - `prompt_tokens`: A reference to a `std::vector<llama_token>` where the resulting tokens from the prompt will be stored.
    - `llama_data`: A constant reference to a `LlamaData` object, which contains context information needed for tokenization.
- **Control Flow**:
    - Check if the current sequence position in the context is zero to determine if this is the first tokenization operation.
    - Calculate the number of tokens required for the prompt by calling `llama_tokenize` with a NULL output buffer and store the negative result in `n_prompt_tokens`.
    - Resize the `prompt_tokens` vector to accommodate the number of tokens calculated.
    - Call `llama_tokenize` again to actually tokenize the prompt into `prompt_tokens`; if this fails, print an error message and return -1.
    - Return the number of tokens in `n_prompt_tokens`.
- **Output**: Returns the number of tokens in the prompt as an integer, or -1 if tokenization fails.
- **Functions called**:
    - [`printe`](#printe)


---
### check\_context\_size<!-- {{#callable:check_context_size}} -->
The `check_context_size` function checks if the current context size plus the number of tokens in a batch exceeds the maximum allowable context size in a given llama context.
- **Inputs**:
    - `ctx`: A constant reference to a `llama_context_ptr` object representing the llama context.
    - `batch`: A constant reference to a `llama_batch` object containing the batch of tokens to be evaluated.
- **Control Flow**:
    - Retrieve the maximum context size `n_ctx` from the llama context using `llama_n_ctx` function.
    - Retrieve the current used context size `n_ctx_used` using `llama_kv_self_seq_pos_max` function with a position of 0.
    - Check if the sum of `n_ctx_used` and `batch.n_tokens` exceeds `n_ctx`.
    - If the context size is exceeded, print a message and return 1.
    - If the context size is not exceeded, return 0.
- **Output**: Returns an integer: 1 if the context size is exceeded, otherwise 0.
- **Functions called**:
    - [`printe`](#printe)


---
### convert\_token\_to\_string<!-- {{#callable:convert_token_to_string}} -->
The `convert_token_to_string` function converts a given token ID from a vocabulary into its corresponding string representation and stores it in a provided string reference.
- **Inputs**:
    - `vocab`: A pointer to a `llama_vocab` structure, representing the vocabulary used for token conversion.
    - `token_id`: A `llama_token` representing the ID of the token to be converted into a string.
    - `piece`: A reference to a `std::string` where the resulting string representation of the token will be stored.
- **Control Flow**:
    - A buffer `buf` of size 256 is declared to hold the string representation of the token.
    - The function `llama_token_to_piece` is called with the provided vocabulary, token ID, buffer, and other parameters to convert the token ID to a string piece.
    - If the conversion function returns a negative value, indicating failure, an error message is printed and the function returns 1.
    - If successful, the buffer content is assigned to the `piece` string, and the function returns 0.
- **Output**: The function returns an integer, 0 on successful conversion and 1 if the conversion fails.
- **Functions called**:
    - [`printe`](#printe)


---
### print\_word\_and\_concatenate\_to\_response<!-- {{#callable:print_word_and_concatenate_to_response}} -->
The function `print_word_and_concatenate_to_response` prints a given string to the standard output and appends it to a response string.
- **Inputs**:
    - `piece`: A constant reference to a `std::string` that represents the word or text to be printed and concatenated.
    - `response`: A reference to a `std::string` where the `piece` will be appended.
- **Control Flow**:
    - The function uses `printf` to print the `piece` to the standard output.
    - It then calls `fflush(stdout)` to ensure the output is immediately flushed to the console.
    - Finally, it appends the `piece` to the `response` string.
- **Output**: The function does not return any value; it modifies the `response` string in place.


---
### generate<!-- {{#callable:generate}} -->
The `generate` function processes a given prompt using a language model to generate a response, which is then returned as a string.
- **Inputs**:
    - `llama_data`: An instance of `LlamaData` containing the model, sampler, and context needed for generation.
    - `prompt`: A `std::string` containing the initial text input for which a response is to be generated.
    - `response`: A `std::string` reference where the generated response will be stored.
- **Control Flow**:
    - Retrieve the vocabulary from the model stored in `llama_data`.
    - Tokenize the input `prompt` using the model's vocabulary and store the tokens in a vector.
    - If tokenization fails, return 1 to indicate an error.
    - Prepare a batch of tokens for processing.
    - Enter a loop to generate tokens until an end-of-generation token is encountered.
    - Check if the context size is sufficient for the current batch; if not, return 1.
    - Decode the current batch of tokens using the model's context; if decoding fails, return 1.
    - Sample the next token using the sampler; if the token is an end-of-generation token, break the loop.
    - Convert the sampled token to a string and append it to the `response`; if conversion fails, return 1.
    - Prepare the next batch using the newly sampled token.
    - After exiting the loop, reset the console color and return 0 to indicate success.
- **Output**: Returns an integer status code: 0 for success and 1 for failure.
- **Functions called**:
    - [`tokenize_prompt`](#tokenize_prompt)
    - [`check_context_size`](#check_context_size)
    - [`printe`](#printe)
    - [`convert_token_to_string`](#convert_token_to_string)
    - [`print_word_and_concatenate_to_response`](#print_word_and_concatenate_to_response)


---
### read\_user\_input<!-- {{#callable:read_user_input}} -->
The `read_user_input` function reads user input from the console, handling different platforms and special input cases, and returns a status code based on the input received.
- **Inputs**:
    - `user_input`: A reference to a string where the user input will be stored.
- **Control Flow**:
    - Check if the environment variable 'LLAMA_PROMPT_PREFIX' is set and use it as the prompt prefix; otherwise, use the default '> ' prefix.
    - On Windows, print the prompt prefix and read a line from standard input into 'user_input'.
    - If the end-of-file (EOF) is reached on Windows, print a newline and return 1.
    - On non-Windows systems, use 'linenoise' to read a line with the prompt prefix and store it in 'user_input'.
    - If 'linenoise' fails to read a line, return 1.
    - Check if 'user_input' is '/bye' and return 1 if true.
    - Check if 'user_input' is empty and return 2 if true.
    - On non-Windows systems, add the input line to the history using 'linenoiseHistoryAdd'.
    - Return 0 if valid input is received.
- **Output**: An integer status code: 0 for successful input, 1 for termination commands or EOF, and 2 for empty input.
- **Functions called**:
    - [`linenoiseHistoryAdd`](linenoise.cpp/linenoise.cpp.driver.md#linenoiseHistoryAdd)


---
### generate\_response<!-- {{#callable:generate_response}} -->
The `generate_response` function generates a response based on a given prompt using LlamaData and outputs it with optional terminal color formatting.
- **Inputs**:
    - `llama_data`: An instance of LlamaData containing model, sampler, and context information necessary for generating a response.
    - `prompt`: A string representing the input prompt for which a response is to be generated.
    - `response`: A reference to a string where the generated response will be stored.
    - `stdout_a_terminal`: A boolean indicating whether the standard output is a terminal, used to determine if color formatting should be applied.
- **Control Flow**:
    - If `stdout_a_terminal` is true, set the response color to yellow using `printf`.
    - Call the [`generate`](#generate) function with `llama_data`, `prompt`, and `response` to generate the response.
    - If [`generate`](#generate) returns a non-zero value, print an error message and return 1 indicating failure.
    - Reset the color to default and print a newline if `stdout_a_terminal` is true.
    - Return 0 indicating successful response generation.
- **Output**: Returns an integer, 0 on success and 1 on failure.
- **Functions called**:
    - [`generate`](#generate)
    - [`printe`](#printe)


---
### apply\_chat\_template\_with\_error\_handling<!-- {{#callable:apply_chat_template_with_error_handling}} -->
The function `apply_chat_template_with_error_handling` applies a chat template to the given data and handles any errors that occur during the process.
- **Inputs**:
    - `tmpls`: A pointer to a `common_chat_templates` structure containing the chat templates to be applied.
    - `llama_data`: A reference to a `LlamaData` object that holds the data to which the chat template will be applied.
    - `append`: A boolean indicating whether to append the generated prompt to existing data.
    - `output_length`: A reference to an integer where the length of the output will be stored.
    - `use_jinja`: A boolean indicating whether to use Jinja templating for the chat template.
- **Control Flow**:
    - Call [`apply_chat_template`](#apply_chat_template) with the provided arguments to apply the chat template.
    - Check if the result of [`apply_chat_template`](#apply_chat_template) is negative, indicating an error.
    - If an error occurred, print an error message and return -1.
    - If no error occurred, update `output_length` with the new length and return 0.
- **Output**: Returns 0 on success, or -1 if an error occurred during the application of the chat template.
- **Functions called**:
    - [`apply_chat_template`](#apply_chat_template)
    - [`printe`](#printe)


---
### handle\_user\_input<!-- {{#callable:handle_user_input}} -->
The `handle_user_input` function processes user input by either using a provided user string or reading input interactively.
- **Inputs**:
    - `user_input`: A reference to a string where the user input will be stored.
    - `user`: A constant reference to a string representing the user input, which, if not empty, will be used directly.
- **Control Flow**:
    - Check if the `user` string is not empty.
    - If `user` is not empty, assign its value to `user_input` and return 0, indicating no need for further input.
    - If `user` is empty, call [`read_user_input`](#read_user_input) to get interactive input and return its result.
- **Output**: Returns an integer, 0 if the user input is directly assigned from `user`, or the result of [`read_user_input`](#read_user_input) if interactive input is needed.
- **Functions called**:
    - [`read_user_input`](#read_user_input)


---
### is\_stdin\_a\_terminal<!-- {{#callable:is_stdin_a_terminal}} -->
The function `is_stdin_a_terminal` checks if the standard input (stdin) is connected to a terminal.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to determine the operating system.
    - On Windows (_WIN32), it retrieves the handle for standard input using `GetStdHandle` and checks if it is a console using `GetConsoleMode`.
    - On non-Windows systems, it uses `isatty` to check if the file descriptor for standard input is a terminal.
- **Output**: The function returns a boolean value indicating whether stdin is a terminal.


---
### is\_stdout\_a\_terminal<!-- {{#callable:is_stdout_a_terminal}} -->
The function `is_stdout_a_terminal` checks if the standard output (stdout) is connected to a terminal.
- **Inputs**: None
- **Control Flow**:
    - The function uses preprocessor directives to determine the operating system.
    - On Windows (_WIN32), it retrieves the handle for the standard output using `GetStdHandle` and checks the console mode with `GetConsoleMode`.
    - On non-Windows systems, it uses `isatty` to check if `STDOUT_FILENO` is a terminal.
- **Output**: The function returns a boolean value indicating whether stdout is a terminal.


---
### get\_user\_input<!-- {{#callable:get_user_input}} -->
The `get_user_input` function repeatedly prompts for user input until valid input is received or a termination condition is met.
- **Inputs**:
    - `user_input`: A reference to a string where the user's input will be stored.
    - `user`: A constant reference to a string representing a predefined user input, which if not empty, bypasses interactive input.
- **Control Flow**:
    - Enter an infinite loop to continuously prompt for user input.
    - Call [`handle_user_input`](#handle_user_input) with `user_input` and `user` to process the input.
    - If [`handle_user_input`](#handle_user_input) returns 1, indicating termination, return 1 to signal the end of input collection.
    - If [`handle_user_input`](#handle_user_input) returns 2, indicating empty input, continue the loop to prompt again.
    - Break the loop if [`handle_user_input`](#handle_user_input) returns any other value, indicating valid input has been received.
- **Output**: Returns an integer, 1 if the input process is terminated by the user, or 0 if valid input is successfully received.
- **Functions called**:
    - [`handle_user_input`](#handle_user_input)


---
### read\_chat\_template\_file<!-- {{#callable:read_chat_template_file}} -->
The `read_chat_template_file` function reads the contents of a specified chat template file and returns it as a string.
- **Inputs**:
    - `chat_template_file`: A string representing the path to the chat template file to be read.
- **Control Flow**:
    - Create a `File` object.
    - Attempt to open the specified file in read mode using the `open` method of the `File` class.
    - If the file cannot be opened, print an error message using [`printe`](#printe) and return an empty string.
    - If the file is successfully opened, read its contents into a string using the `to_string` method of the `File` class.
    - Return the string containing the file's contents.
- **Output**: A string containing the contents of the chat template file, or an empty string if the file could not be opened.
- **Functions called**:
    - [`printe`](#printe)


---
### process\_user\_message<!-- {{#callable:process_user_message}} -->
The `process_user_message` function processes a user's input message, generates a response using a chat template, and updates the chat history accordingly.
- **Inputs**:
    - `opt`: An instance of the `Opt` class containing configuration options such as user name and template usage.
    - `user_input`: A string representing the input message from the user.
    - `llama_data`: A reference to a `LlamaData` object that holds the model, context, and message data.
    - `chat_templates`: A pointer to a `common_chat_templates` object used for applying chat templates.
    - `prev_len`: An integer reference representing the previous length of the formatted message data.
    - `stdout_a_terminal`: A boolean indicating whether the standard output is a terminal.
- **Control Flow**:
    - Add the user's message to the `llama_data` message list, using either the provided `user_input` or the `opt.user` if available.
    - Apply the chat template to the current messages, updating the formatted message data and storing the new length in `new_len`.
    - If applying the chat template fails, return 1 to indicate an error.
    - Extract the prompt from the formatted message data using the range from `prev_len` to `new_len`.
    - Generate a response based on the extracted prompt and store it in the `response` variable.
    - If generating the response fails, return 1 to indicate an error.
    - If `opt.user` is not empty, return 2 to indicate a special condition where the user input is predefined.
    - Add the generated response as an 'assistant' message to the `llama_data` message list.
    - Reapply the chat template to update the formatted message data for the next interaction, updating `prev_len`.
    - If reapplying the chat template fails, return 1 to indicate an error.
    - Return 0 to indicate successful processing of the user message.
- **Output**: An integer indicating the result of processing: 0 for success, 1 for error, and 2 for a special condition when `opt.user` is predefined.
- **Functions called**:
    - [`add_message`](#add_message)
    - [`apply_chat_template_with_error_handling`](#apply_chat_template_with_error_handling)
    - [`generate_response`](#generate_response)


---
### chat\_loop<!-- {{#callable:chat_loop}} -->
The `chat_loop` function manages an interactive chat session, processing user inputs and generating responses using a language model.
- **Inputs**:
    - `llama_data`: An instance of `LlamaData` containing the model, context, and other necessary data for the chat session.
    - `opt`: An instance of `Opt` containing configuration options such as chat template file, user input, and other parameters.
- **Control Flow**:
    - Initialize `prev_len` to 0 and resize `llama_data.fmtted` to the context size of the model.
    - If a chat template file is specified in `opt`, read its content into `chat_template`.
    - Initialize `chat_templates` using the model and the chat template.
    - Determine if the standard output is a terminal and store the result in `stdout_a_terminal`.
    - Enter an infinite loop to continuously process user input.
    - Call [`get_user_input`](#get_user_input) to retrieve user input; if it returns 1, exit the loop and return 0.
    - Call [`process_user_message`](#process_user_message) to process the user input and generate a response; handle return values to either continue, break, or return 1.
- **Output**: Returns an integer status code: 0 for normal termination, 1 for errors, and breaks the loop if the user input indicates termination.
- **Functions called**:
    - [`read_chat_template_file`](#read_chat_template_file)
    - [`is_stdout_a_terminal`](#is_stdout_a_terminal)
    - [`get_user_input`](#get_user_input)
    - [`process_user_message`](#process_user_message)


---
### log\_callback<!-- {{#callable:log_callback}} -->
The `log_callback` function logs messages based on the verbosity level set in the `Opt` object or if the log level is an error.
- **Inputs**:
    - `level`: An enumeration value of type `ggml_log_level` indicating the severity of the log message.
    - `text`: A constant character pointer to the log message text.
    - `p`: A void pointer to an `Opt` object, which contains configuration options including verbosity.
- **Control Flow**:
    - Cast the void pointer `p` to an `Opt` pointer.
    - Check if the `verbose` flag in the `Opt` object is true or if the log level is `GGML_LOG_LEVEL_ERROR`.
    - If either condition is true, call the [`printe`](#printe) function to print the log message to standard error.
- **Output**: The function does not return a value; it performs logging as a side effect.
- **Functions called**:
    - [`printe`](#printe)


---
### read\_pipe\_data<!-- {{#callable:read_pipe_data}} -->
The `read_pipe_data` function reads all data from the standard input stream (`std::cin`) and returns it as a string.
- **Inputs**: None
- **Control Flow**:
    - Create an `std::ostringstream` object named `result`.
    - Use the `<<` operator to read all data from `std::cin` into `result`.
    - Return the string representation of the data stored in `result` using `result.str()`.
- **Output**: A `std::string` containing all the data read from the standard input stream.


---
### ctrl\_c\_handling<!-- {{#callable:ctrl_c_handling}} -->
The `ctrl_c_handling` function sets up a signal handler for the SIGINT signal (Ctrl+C) to ensure graceful termination of the program on both Unix-like and Windows systems.
- **Inputs**: None
- **Control Flow**:
    - The function checks if the code is being compiled for a Unix-like system or macOS using preprocessor directives.
    - If on a Unix-like system or macOS, it initializes a `sigaction` structure, sets its handler to [`sigint_handler`](#sigint_handler), clears the signal mask, sets flags to 0, and registers it to handle SIGINT using `sigaction`.
    - If on a Windows system, it defines a lambda function `console_ctrl_handler` to handle the `CTRL_C_EVENT` by calling [`sigint_handler`](#sigint_handler) and returns true, then registers this handler using `SetConsoleCtrlHandler`.
- **Output**: The function does not return any value.
- **Functions called**:
    - [`sigint_handler`](#sigint_handler)


---
### main<!-- {{#callable:main}} -->
The `main` function initializes the application, processes command-line arguments, sets up logging, and enters a chat loop with the Llama model.
- **Inputs**:
    - `argc`: The number of command-line arguments passed to the program.
    - `argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Call `ctrl_c_handling()` to set up signal handling for Ctrl+C interrupts.
    - Create an `Opt` object and initialize it with command-line arguments using `opt.init(argc, argv)`.
    - Check the return value of `opt.init`; if it is 2, return 0 to exit normally, or return 1 to indicate an error.
    - Check if standard input is not a terminal; if so, append any piped data to `opt.user`.
    - Set up logging with `llama_log_set(log_callback, &opt)`.
    - Create a `LlamaData` object and initialize it with `llama_data.init(opt)`; return 1 if initialization fails.
    - Enter the chat loop with `chat_loop(llama_data, opt)`; return 1 if the chat loop fails.
    - Return 0 to indicate successful execution.
- **Output**: The function returns an integer status code: 0 for successful execution, 1 for errors, and 2 for help display.
- **Functions called**:
    - [`ctrl_c_handling`](#ctrl_c_handling)
    - [`is_stdin_a_terminal`](#is_stdin_a_terminal)
    - [`read_pipe_data`](#read_pipe_data)
    - [`chat_loop`](#chat_loop)


