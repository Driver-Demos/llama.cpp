# Purpose
This C++ source code file is an executable program designed to tokenize a given text prompt using a specified model. The program reads command-line arguments to determine the model file path, the source of the prompt (either from a file, standard input, or directly from an argument), and various options that control the tokenization process. The code includes functionality to handle different character encodings, particularly on Windows, ensuring that non-ASCII characters are processed correctly. It also provides options to control the output format, such as printing only numerical token IDs or including additional information like the total number of tokens.

The main technical components of the code include functions for reading command-line arguments, handling file input, and managing character encoding conversions. The program uses the `llama` library to load the model and perform tokenization, and it provides a detailed usage guide for users through the [`print_usage_information`](#print_usage_information) function. The code is structured to handle various user inputs and configurations robustly, with error checking and informative error messages. It is intended to be run as a standalone application, as indicated by the presence of a [`main`](#main) function, and it does not define public APIs or external interfaces for use by other programs.
# Imports and Dependencies

---
- `common.h`
- `llama.h`
- `cstdio`
- `cstring`
- `fstream`
- `string`
- `vector`
- `iostream`
- `windows.h`
- `shellapi.h`


# Functions

---
### print\_usage\_information<!-- {{#callable:print_usage_information}} -->
The `print_usage_information` function outputs the usage instructions and available options for a command-line tokenization program.
- **Inputs**:
    - `argv0`: A constant character pointer representing the name of the program, typically the first argument in the command-line arguments array.
- **Control Flow**:
    - The function uses `printf` to output a formatted string that includes the program name and a list of options.
    - It describes the purpose of the program, which is to tokenize a prompt using a specified model and print the tokens.
    - The function lists various command-line options, explaining their purpose and usage.
- **Output**: The function does not return any value; it outputs information directly to the standard output.


---
### llama\_log\_callback\_null<!-- {{#callable:llama_log_callback_null}} -->
The `llama_log_callback_null` function is a no-operation logging callback that ignores all its input parameters.
- **Inputs**:
    - `level`: The logging level, represented by the `ggml_log_level` type, which is ignored in this function.
    - `text`: A pointer to a constant character string representing the log message, which is ignored in this function.
    - `user_data`: A pointer to user-defined data, which is ignored in this function.
- **Control Flow**:
    - The function takes three parameters: `level`, `text`, and `user_data`, all of which are marked as unused using the `(void)` cast to suppress compiler warnings about unused parameters.
    - The function body contains no operations, effectively making it a no-op function.
- **Output**: This function does not produce any output or perform any operations.


---
### read\_prompt\_from\_file<!-- {{#callable:read_prompt_from_file}} -->
The `read_prompt_from_file` function reads the contents of a file specified by a file path into a string and indicates success or failure through a reference parameter.
- **Inputs**:
    - `filepath`: A constant character pointer representing the path to the file to be read.
    - `success`: A reference to a boolean variable that will be set to true if the file is successfully read, and false otherwise.
- **Control Flow**:
    - Initialize the success reference to false.
    - Attempt to open the file specified by filepath in binary mode using an ifstream object.
    - If the file cannot be opened, print an error message to stderr and return an empty string.
    - Read the entire contents of the file into a stringstream buffer.
    - If reading the file fails, print an error message to stderr and return an empty string.
    - Set the success reference to true if the file is read successfully.
    - Return the contents of the buffer as a string.
- **Output**: Returns a string containing the contents of the file if read successfully, otherwise returns an empty string.


---
### ingest\_args<!-- {{#callable:ingest_args}} -->
The `ingest_args` function converts command-line arguments into a vector of UTF-8 encoded strings, handling character encoding differences on Windows.
- **Inputs**:
    - `raw_argc`: The number of command-line arguments passed to the program.
    - `raw_argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Initialize an empty vector of strings named `argv`.
    - Check if the platform is Windows using a preprocessor directive.
    - On Windows, retrieve command-line arguments in wide character format using `GetCommandLineW` and `CommandLineToArgvW`.
    - Convert each wide character argument to a UTF-8 encoded string using `WideCharToMultiByte` and add it to the `argv` vector.
    - Free the memory allocated for wide character arguments using `LocalFree`.
    - On non-Windows platforms, directly copy the `raw_argv` arguments into the `argv` vector.
    - Assert that the number of arguments (`argc`) matches the size of the `argv` vector.
    - Return the `argv` vector.
- **Output**: A vector of strings containing the command-line arguments encoded in UTF-8.


---
### write\_utf8\_cstr\_to\_stdout<!-- {{#callable:write_utf8_cstr_to_stdout}} -->
The function `write_utf8_cstr_to_stdout` writes a UTF-8 encoded C-style string to the standard output, handling Windows-specific console output requirements and detecting invalid UTF-8 sequences.
- **Inputs**:
    - `str`: A constant character pointer representing the UTF-8 encoded C-style string to be written to standard output.
    - `invalid_utf8`: A reference to a boolean variable that will be set to true if the input string contains invalid UTF-8 sequences.
- **Control Flow**:
    - Initialize `invalid_utf8` to false.
    - Check if the platform is Windows using the `_WIN32` macro.
    - On Windows, obtain the console handle using `GetStdHandle` and check if it is valid using `GetConsoleMode`.
    - If the console handle is invalid or the console mode cannot be retrieved, print the string using `printf` and return.
    - Check if the input string is empty; if so, return without doing anything.
    - Use `MultiByteToWideChar` to determine the length needed for the wide character representation of the string.
    - If `MultiByteToWideChar` returns zero, check the error code using `GetLastError`.
    - If the error is `ERROR_NO_UNICODE_TRANSLATION`, set `invalid_utf8` to true and print the string as a sequence of hexadecimal byte values enclosed in angle brackets, then return.
    - If `MultiByteToWideChar` fails unexpectedly, abort the program using `GGML_ABORT`.
    - Allocate memory for the wide character string, convert the input string using `MultiByteToWideChar`, and write it to the console using `WriteConsoleW`.
    - Free the allocated memory for the wide character string.
    - On non-Windows platforms, simply print the string using `printf`.
- **Output**: The function does not return a value but writes the input string to standard output and sets the `invalid_utf8` flag to true if the string contains invalid UTF-8 sequences.


---
### main<!-- {{#callable:main}} -->
The `main` function processes command-line arguments to configure and execute a tokenization task using a specified model and prompt, then outputs the resulting tokens.
- **Inputs**:
    - `raw_argc`: The number of command-line arguments passed to the program.
    - `raw_argv`: An array of C-style strings representing the command-line arguments.
- **Control Flow**:
    - Convert command-line arguments to a vector of strings using [`ingest_args`](#ingest_args) to handle encoding issues, especially on Windows.
    - Check if the number of arguments is less than or equal to 1, and if so, print usage information and exit with code 1.
    - Initialize variables to store command-line options and flags, such as `printing_ids`, `no_bos`, `no_escape`, etc.
    - Iterate over the command-line arguments to parse and set the appropriate flags and options, handling errors for invalid or repeated arguments.
    - Perform sanity checks on the parsed arguments to ensure required options are set and mutually exclusive options are not both set.
    - Determine the source of the prompt (file, argument, or stdin) and read it accordingly, handling errors if reading fails.
    - Initialize the tokenization backend and load the model from the specified file, checking for errors in loading.
    - Initialize the context for tokenization using the loaded model, checking for errors in context creation.
    - If reading from stdin, capture the input after model loading to ensure early exit if model loading fails.
    - Configure tokenization options such as adding a BOS token, parsing special tokens, and escaping input based on flags.
    - Tokenize the prompt using the configured options and the model's vocabulary.
    - Output the tokens either as numerical IDs or as strings, depending on the `printing_ids` flag, handling UTF-8 encoding issues.
    - If the `show_token_count` flag is set, print the total number of tokens.
    - Free the allocated resources for the model and context before exiting.
- **Output**: Returns 0 on successful execution, or 1 if an error occurs during argument parsing, model loading, or tokenization.
- **Functions called**:
    - [`ingest_args`](#ingest_args)
    - [`print_usage_information`](#print_usage_information)
    - [`read_prompt_from_file`](#read_prompt_from_file)
    - [`llama_backend_init`](../../src/llama.cpp.driver.md#llama_backend_init)
    - [`llama_model_default_params`](../../src/llama-model.cpp.driver.md#llama_model_default_params)
    - [`llama_context_default_params`](../../src/llama-context.cpp.driver.md#llama_context_default_params)
    - [`write_utf8_cstr_to_stdout`](#write_utf8_cstr_to_stdout)


